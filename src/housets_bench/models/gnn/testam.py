"""TESTAM — Time-Enhanced Spatio-temporal Attention Model with Mixture of
experts (Lee & Ko, ICLR 2024).

Reference: https://github.com/HyunWookL/TESTAM  (arXiv:2403.02600)

Confirmed from source (``model.py``/``engine.py``): 3 experts —
``TemporalModel`` (identity/no-graph: node-identity embedding + pure
temporal attention), ``STModel`` (a *learned*, not external, static
adjacency built from a memory bank via two linear projections, then GCN +
temporal attention), ``AttentionModel`` (fully dynamic: alternating
spatial/temporal attention, no fixed graph at all) — gated by a shared
memory bank queried via cosine similarity by both the raw input and each
expert's own hidden state, with **hard top-1 routing at both train and eval
time** (not a dense weighted mixture), plus a 2-term routing loss
("worst avoidance" + "best choice") added after a warmup period during
which every expert is trained directly against the ground truth
(an "ind_loss" term) so routing has something informative to route toward.
None of the 3 experts consult an external adjacency — ``STModel``'s
adjacency is fully self-learned, same category as this registry's
``agcrn.py`` — so ``requires_graph = False``.

**Caveat, stated plainly**: the macro-architecture above (3 experts, the
gate's shared-memory-bank + cosine-similarity design, hard top-1 routing,
warmup-then-routing-losses training schedule) is confirmed from the actual
repo source. The *exact* tensor-level formula for combining the gate's
input-query and hidden-query similarities into one per-expert score, and
the exact "uncertainty" weighting constants inside the two routing-loss
terms, were not pinned down to literal source line-by-line — this port uses
a documented, reasonable reconstruction of both (see ``MemoryGate.forward``
and ``TESTAMForecaster._graph_forward_train``), not a byte-identical copy.
Hyperparameters not found in the fetched source (``memory_size``) use a
documented, reasonable default. Time-of-day/day-of-week are available via
``GraphWindowDataset``'s ``x_mark`` (same mechanism as
``staeformer.py``/``stid.py``) but not wired in here, since the original
repo concatenates them as a plain input channel upstream of the model
rather than inside it — add a ``time_of_day`` feature column via the
traffic data loader's ``feature_cols`` instead (same convention documented
in ``dcrnn.py``/``graph_wavenet.py``/``mtgnn.py``).
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from housets_bench.models.gnn.gnn_forecaster import GNNForecasterBase
from housets_bench.models.gnn.stgformer import GraphPropagate
from housets_bench.models.registry import register


class _TemporalAttention(nn.Module):
    """Per-node self-attention over the lookback window."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, L, N, D = h.shape
        h_bn = h.permute(0, 2, 1, 3).reshape(B * N, L, D)
        out, _ = self.attn(h_bn, h_bn, h_bn)
        h_bn = self.norm(h_bn + self.drop(out))
        return h_bn.reshape(B, N, L, D).permute(0, 2, 1, 3)


class _SpatialAttention(nn.Module):
    """Per-timestep self-attention over nodes — dynamic, no fixed graph."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, L, N, D = h.shape
        h_bl = h.reshape(B * L, N, D)
        out, _ = self.attn(h_bl, h_bl, h_bl)
        h_bl = self.norm(h_bl + self.drop(out))
        return h_bl.reshape(B, L, N, D)


class TemporalExpert(nn.Module):
    """Identity expert: node-identity embedding + pure temporal attention, no graph."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float, n_nodes: int, node_emb_dim: int) -> None:
        super().__init__()
        self.node_emb = nn.Parameter(torch.randn(n_nodes, node_emb_dim) * 0.1)
        self.proj_in = nn.Linear(d_model + node_emb_dim, d_model)
        self.layers = nn.ModuleList([_TemporalAttention(d_model, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, L, N, _ = h.shape
        ne = self.node_emb.view(1, 1, N, -1).expand(B, L, -1, -1)
        h = self.proj_in(torch.cat([h, ne], dim=-1))
        for layer in self.layers:
            h = layer(h)
        return h


class _StaticAdaptiveGraph(nn.Module):
    """Static (per-dataset, not per-instance) learned adjacency from a memory
    bank: ``n1 = We1(memory)``, ``n2 = We2(memory)``, ``A = softmax(relu(n1 @ n2.T))``
    — confirmed formula, no external graph consulted."""

    def __init__(self, n_nodes: int, emb_dim: int) -> None:
        super().__init__()
        self.memory = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
        self.we1 = nn.Linear(emb_dim, emb_dim)
        self.we2 = nn.Linear(emb_dim, emb_dim)

    def forward(self) -> torch.Tensor:
        n1 = self.we1(self.memory)
        n2 = self.we2(self.memory)
        return F.softmax(F.relu(n1 @ n2.t()), dim=-1)


class STExpert(nn.Module):
    """Static expert: memory-bank adjacency -> GCN (order=2) -> temporal attention."""

    def __init__(
        self, d_model: int, n_heads: int, n_layers: int, dropout: float, n_nodes: int, emb_dim: int, order: int = 2,
    ) -> None:
        super().__init__()
        self.graph = _StaticAdaptiveGraph(n_nodes, emb_dim)
        self.prop = GraphPropagate(d_model, order, dropout)
        self.layers = nn.ModuleList([_TemporalAttention(d_model, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        A = self.graph()
        h = self.prop(h, A)
        for layer in self.layers:
            h = layer(h)
        return h


class AttentionExpert(nn.Module):
    """Dynamic expert: alternating temporal/spatial attention, no fixed graph at all."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.temporal_layers = nn.ModuleList([_TemporalAttention(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.spatial_layers = nn.ModuleList([_SpatialAttention(d_model, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        for t_layer, s_layer in zip(self.temporal_layers, self.spatial_layers):
            h = t_layer(h)
            h = s_layer(h)
        return h


class MemoryGate(nn.Module):
    """Shared memory bank queried by cosine similarity from the raw input and
    each expert's own hidden state, producing a per-node, per-expert score.

    Confirmed from source: a shared memory bank + cosine-similarity query
    from both the input and each expert's hidden state. The exact formula
    combining the two similarities into one score (here: sum of the
    max-over-memory-slot similarity from each) is a documented
    reconstruction, not a verified line-by-line copy — see module docstring.
    """

    def __init__(self, d_model: int, memory_size: int, mem_hid: int, n_experts: int = 3) -> None:
        super().__init__()
        self.memory = nn.Parameter(torch.randn(memory_size, mem_hid) * 0.1)
        self.input_query = nn.Linear(d_model, mem_hid)
        self.hid_query = nn.ModuleList([nn.Linear(d_model, mem_hid) for _ in range(n_experts)])
        self.n_experts = n_experts

    def _similarity(self, q: torch.Tensor) -> torch.Tensor:
        # q: [..., mem_hid] -> [..., memory_size]
        q_n = F.normalize(q, dim=-1)
        m_n = F.normalize(self.memory, dim=-1)
        return q_n @ m_n.t()

    def forward(self, x_summary: torch.Tensor, expert_hiddens: List[torch.Tensor]) -> torch.Tensor:
        # x_summary, each expert_hiddens[i]: [B, N, d_model] (mean-over-L summaries)
        in_sim = self._similarity(self.input_query(x_summary)).amax(dim=-1)  # [B, N]
        scores = []
        for i, h in enumerate(expert_hiddens):
            hid_sim = self._similarity(self.hid_query[i](h)).amax(dim=-1)  # [B, N]
            scores.append(in_sim + hid_sim)
        gate_logits = torch.stack(scores, dim=-1)  # [B, N, n_experts]
        return F.softmax(gate_logits, dim=-1)


class TESTAMNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        pred_len: int,
        n_nodes: int,
        *,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        node_emb_dim: int = 16,
        st_emb_dim: int = 16,
        memory_size: int = 20,
        mem_hid: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.pred_len = int(pred_len)
        self.out_dim = int(out_dim)

        self.in_proj = nn.Linear(int(input_dim), int(d_model))
        self.temporal_expert = TemporalExpert(d_model, n_heads, n_layers, dropout, n_nodes, node_emb_dim)
        self.st_expert = STExpert(d_model, n_heads, n_layers, dropout, n_nodes, st_emb_dim)
        self.attention_expert = AttentionExpert(d_model, n_heads, n_layers, dropout)
        self.gate = MemoryGate(d_model, memory_size, mem_hid, n_experts=3)

        self.out_proj = nn.ModuleList([nn.Linear(d_model, self.pred_len * self.out_dim) for _ in range(3)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Returns (routed_pred [B,pred_len,N,Dy], gate [B,N,3], expert_preds: list of 3 [B,pred_len,N,Dy])."""
        B, L, N, _ = x.shape
        h0 = self.in_proj(x)  # [B, L, N, d_model]

        experts = [self.temporal_expert, self.st_expert, self.attention_expert]
        hiddens = [expert(h0) for expert in experts]  # each [B, L, N, d_model]
        last_hidden = [h[:, -1, :, :] for h in hiddens]  # [B, N, d_model] -- used for routing + prediction
        x_summary = h0.mean(dim=1)  # [B, N, d_model]

        gate = self.gate(x_summary, last_hidden)  # [B, N, 3]

        expert_preds = []
        for i, h_last in enumerate(last_hidden):
            out = self.out_proj[i](h_last)  # [B, N, pred_len*out_dim]
            expert_preds.append(out.view(B, N, self.pred_len, self.out_dim).permute(0, 2, 1, 3))

        # hard top-1 routing, per node (confirmed from source — same at train and eval)
        top1 = gate.argmax(dim=-1)  # [B, N]
        stacked = torch.stack(expert_preds, dim=0)  # [3, B, pred_len, N, Dy]
        idx = top1.reshape(1, B, 1, N, 1).expand(1, B, self.pred_len, N, self.out_dim)
        routed = stacked.gather(0, idx).squeeze(0)  # [B, pred_len, N, Dy]

        return routed, gate, expert_preds


@register("testam")
class TESTAMForecaster(GNNForecasterBase):
    """TESTAM forecaster: 3-expert mixture (identity / learned-static-graph /
    dynamic-attention) with hard top-1 routing over a shared memory-bank gate."""

    name: str = "testam"
    requires_graph: bool = False
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    node_emb_dim: int = 16
    st_emb_dim: int = 16
    memory_size: int = 20
    mem_hid: int = 32
    dropout: float = 0.3
    warmup_frac: float = 0.2       # fraction of total training steps using ind_loss only
    routing_loss_weight: float = 0.5

    def _build_net(self, bundle, n_nodes, *, A_norm, device):
        return TESTAMNet(
            input_dim=len(bundle.x_cols),
            out_dim=len(bundle.y_cols),
            pred_len=int(bundle.raw.spec.pred_len),
            n_nodes=n_nodes,
            d_model=int(self.d_model),
            n_heads=int(self.n_heads),
            n_layers=int(self.n_layers),
            node_emb_dim=int(self.node_emb_dim),
            st_emb_dim=int(self.st_emb_dim),
            memory_size=int(self.memory_size),
            mem_hid=int(self.mem_hid),
            dropout=float(self.dropout),
        )

    def _graph_forward(self, net, x):
        routed, _gate, _expert_preds = net(x)
        return routed

    def _graph_forward_train(self, net, x, y_true, progress):
        routed, gate, expert_preds = net(x)

        # "ind_loss": every expert trained directly against ground truth, so
        # routing has something informative to route toward from step one.
        ind_losses = torch.stack([F.mse_loss(ep, y_true, reduction="none").mean(dim=(1, 3)) for ep in expert_preds], dim=-1)
        ind_loss = ind_losses.mean()

        if progress < float(self.warmup_frac):
            return routed, ind_loss

        # Routing-encouragement terms, confirmed structure (worst-avoidance +
        # best-choice, each weighted by an "uncertainty" term) -- see module
        # docstring for what's a documented reconstruction vs. confirmed.
        best_idx = ind_losses.argmin(dim=-1)   # [B, N] -- lowest per-node error
        worst_idx = ind_losses.argmax(dim=-1)  # [B, N] -- highest per-node error
        err_min = ind_losses.min(dim=-1).values
        err_max = ind_losses.max(dim=-1).values
        uncertainty = (err_max - err_min) / (err_max + 1e-8)  # in [0,1): how separable the experts are

        eps = 1e-8
        gate_best = gate.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1).clamp(min=eps)
        gate_worst = gate.gather(-1, worst_idx.unsqueeze(-1)).squeeze(-1).clamp(max=1 - eps)

        w = float(self.routing_loss_weight)
        best_choice_loss = (-w * uncertainty * torch.log(gate_best)).mean()
        worst_avoidance_loss = (-w * (1.0 - uncertainty) * torch.log((1.0 - gate_worst).clamp(min=eps))).mean()

        aux_loss = ind_loss + best_choice_loss + worst_avoidance_loss
        return routed, aux_loss
