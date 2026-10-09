"""STExplainer — Explainable Spatio-Temporal Graph Neural Networks (HKUDS, CIKM 2023).

Reference: https://github.com/HKUDS/STExplainer  (arXiv:2310.17149)

Faithful port of the actually-shipped model — verified from source, not from the
paper's prose alone. The repo ships three model classes (``STGIB``, ``STGSAT``,
``STGAT``); the one shipped config/checkpoint trains **STGSAT**: an STID-style
backbone (learned per-node identity embedding, raw-residual branch) wrapped
around a **GSAT** (Graph Stochastic Attention) information-bottleneck module,
applied **twice** — once over the real spatial adjacency (nodes = regions), once
over a synthetic *complete graph* across the lookback window (nodes = the ``L``
timesteps). Each GSAT pass:
  1. computes plain (unmasked) node embeddings via a 2-layer GAT,
  2. an ``ExtractorMLP`` turns each node-pair embedding into an edge logit,
  3. a binary-concrete (Gumbel-sigmoid) relaxation samples a stochastic edge gate
     ``att`` from those logits (deterministic ``sigmoid`` at eval time),
  4. a second masked 2-layer GAT (``alpha *= att``) produces the branch's output
     embedding,
  5. ``KL(Bernoulli(att) ‖ Bernoulli(r))`` regularizes the gate towards a target
     sparsity ``r``, annealed 0.9→0.5 over the first ~40% of training — this is
     the actual "Graph Information Bottleneck" term, and it is **structure-only**
     (which edges matter), not a feature-channel bottleneck.

**Extension beyond the paper** (the real model has no feature-channel mechanism
at all — its raw input is collapsed by one ``Linear`` before any GIB module
runs): a **third GSAT pass over a complete graph across the ``Dx`` input feature
channels**, applied to the *raw* input (channel identity is lost once
``start_fc`` mixes channels together, so this branch must operate before that
mixing). Mirrors exactly how the paper already treats time as a complete-graph
GSAT pass — same mechanism, new axis. This is our own addition, not present in
HKUDS/STExplainer; kept clearly separate via its own weight (``beta_feat``) so
it can be zeroed out to recover the paper-faithful 2-axis model if wanted.

``T_i_D``/``D_i_W`` (time-of-day / day-of-week embeddings, wall-clock-periodicity
features in the source) are dropped — this benchmark's datasets are monthly.

Each GSAT pass's edge-attention gate is exposed per-instance via
:meth:`STExplainerForecaster.explain` (spatial + temporal + feature attention,
plus node/time/feature importance derived by aggregating each axis's attention
over incident edges) — the model's real, causally-load-bearing explanation
output, not a post-hoc add-on.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from st_numeric_baselines.models.gnn.gnn_forecaster import GNNForecasterBase
from st_numeric_baselines.models.registry import register


class GATLayer(nn.Module):
    """Dense multi-head GAT-style attention (no PyG — consistent with every other
    GNN module in this codebase, all of which use dense/masked attention).

    ``e_ij = LeakyReLU(a_src·Wh_i + a_dst·Wh_j)``, softmaxed over neighbors per a
    dense adjacency mask; an optional ``edge_atten`` gate multiplies into the
    softmaxed weights before aggregation (this is how GSAT's learned edge gate
    gets injected into ordinary GAT attention).
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if out_dim % n_heads != 0:
            raise ValueError(f"out_dim={out_dim} must be divisible by n_heads={n_heads}")
        self.n_heads = int(n_heads)
        self.head_dim = out_dim // n_heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.randn(self.n_heads, self.head_dim) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(self.n_heads, self.head_dim) * 0.1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, adj_mask: torch.Tensor, edge_atten: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, M, in_dim]; adj_mask: [M, M] (1 where an edge exists, incl. self-loops)
        # edge_atten: optional [B, M, M] in (0,1), gates the softmaxed attention
        B, M, _ = x.shape
        h = self.W(x).view(B, M, self.n_heads, self.head_dim)  # [B,M,H,Dh]

        src_score = (h * self.a_src).sum(-1)  # [B,M,H]
        dst_score = (h * self.a_dst).sum(-1)  # [B,M,H]
        e = self.leaky_relu(src_score.unsqueeze(2) + dst_score.unsqueeze(1))  # [B,M,M,H]

        mask = adj_mask.to(dtype=torch.bool).view(1, M, M, 1)
        e = e.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(e, dim=2)
        alpha = torch.nan_to_num(alpha, nan=0.0)  # fully-isolated rows -> all -inf -> softmax nan -> 0
        if edge_atten is not None:
            alpha = alpha * edge_atten.unsqueeze(-1)  # [B,M,M,1] broadcasts over heads
        alpha = self.drop(alpha)

        out = torch.einsum("bijh,bjhd->bihd", alpha, h).reshape(B, M, self.n_heads * self.head_dim)
        return out, alpha.mean(dim=-1)  # attention averaged over heads: [B,M,M]


class GSATModule(nn.Module):
    """Graph Stochastic Attention: a stochastic, KL-regularized edge gate learned
    from node embeddings, used to mask a second GAT pass producing the branch's
    output embedding. Applied identically regardless of what the graph's "nodes"
    represent (regions, timesteps, or feature channels) — only ``adj_mask``/dims differ.
    """

    def __init__(self, dim: int, n_heads_l1: int, n_heads_l2: int, dropout: float, extractor_dropout: float) -> None:
        super().__init__()
        self.gat1 = GATLayer(dim, dim, n_heads_l1, dropout)
        self.gat2 = GATLayer(dim, dim, n_heads_l2, dropout)
        self.extractor = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(), nn.Dropout(extractor_dropout), nn.Linear(dim, 1)
        )

    def forward(
        self, x: torch.Tensor, adj_mask: torch.Tensor, *, r: float, training: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, M, dim]; adj_mask: [M, M]
        B, M, _ = x.shape
        emb, _ = self.gat1(x, adj_mask)  # unmasked "get_emb" pass
        emb = F.elu(emb)

        pair = torch.cat(
            [emb.unsqueeze(2).expand(-1, -1, M, -1), emb.unsqueeze(1).expand(-1, M, -1, -1)], dim=-1
        )  # [B,M,M,2*dim]
        logits = self.extractor(pair).squeeze(-1)  # [B,M,M]
        att = self._concrete_sample(logits, training)
        att = att * adj_mask.unsqueeze(0)  # restrict the gate to real edges only

        out_emb, _ = self.gat2(emb, adj_mask, edge_atten=att)
        info_loss = self._info_loss(att, adj_mask, r)
        return out_emb, att, info_loss

    @staticmethod
    def _concrete_sample(logits: torch.Tensor, training: bool, temp: float = 1.0) -> torch.Tensor:
        if training:
            eps = 1e-10
            u = torch.rand_like(logits).clamp(eps, 1 - eps)
            gumbel = torch.log(u) - torch.log(1 - u)
            return torch.sigmoid((logits + gumbel) / temp)
        return torch.sigmoid(logits)

    @staticmethod
    def _info_loss(att: torch.Tensor, adj_mask: torch.Tensor, r: float, eps: float = 1e-6) -> torch.Tensor:
        att_c = att.clamp(eps, 1 - eps)
        r = float(min(max(r, eps), 1 - eps))
        kl = att_c * torch.log(att_c / r + eps) + (1 - att_c) * torch.log((1 - att_c) / (1 - r + eps) + eps)
        mask = adj_mask.unsqueeze(0).expand_as(kl)
        return (kl * mask).sum() / mask.sum().clamp_min(1.0)


class _MLPRes(nn.Module):
    """Residual 2-layer MLP block (matches source's ``MLP_res``)."""

    def __init__(self, dim: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(F.relu(self.fc1(x)))
        return x + self.fc2(h)


class STExplainerNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        pred_len: int,
        n_nodes: int,
        seq_len: int,
        *,
        d_model: int = 16,
        d_model_spat: int = 64,
        d_model_temp: int = 64,
        d_model_feat: int = 32,
        node_emb_dim: int = 32,
        n_heads_l1: int = 4,
        n_heads_l2: int = 1,
        dropout: float = 0.15,
        extractor_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.pred_len = int(pred_len)
        self.out_dim = int(out_dim)
        self.seq_len = int(seq_len)
        self.n_nodes = int(n_nodes)
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self._r = float(0.5)  # updated via set_r() before each forward during training

        self.start_fc = nn.Linear(self.input_dim, self.d_model)

        # spatial branch: nodes = regions, over the real supplied adjacency
        self.trans_spat = nn.Linear(self.seq_len * self.d_model, d_model_spat)
        self.inv_trans_spat = nn.Linear(d_model_spat, self.seq_len * self.d_model)
        self.spat_gsat = GSATModule(d_model_spat, n_heads_l1, n_heads_l2, dropout, extractor_dropout)

        # temporal branch: nodes = the L lookback timesteps, complete graph K_L
        self.trans_temp = nn.Linear(self.n_nodes * self.d_model, d_model_temp)
        self.inv_trans_temp = nn.Linear(d_model_temp, self.n_nodes * self.d_model)
        self.temp_gsat = GSATModule(d_model_temp, n_heads_l1, n_heads_l2, dropout, extractor_dropout)
        self.register_buffer("temp_adj", torch.ones(self.seq_len, self.seq_len))

        # feature branch (our extension): nodes = the Dx input channels, complete graph K_Dx
        self.trans_feat = nn.Linear(self.seq_len * self.n_nodes, d_model_feat)
        self.inv_trans_feat = nn.Linear(d_model_feat, self.seq_len * self.n_nodes)
        self.feat_gsat = GSATModule(d_model_feat, n_heads_l1, n_heads_l2, dropout, extractor_dropout)
        self.register_buffer("feat_adj", torch.ones(self.input_dim, self.input_dim))

        # fusion: STID-style raw-residual + branch embeddings + learned node identity
        self.node_emb = nn.Parameter(torch.randn(self.n_nodes, node_emb_dim) * 0.1)
        self.res_proj = nn.Linear(self.seq_len * self.d_model, self.d_model)
        self.temp_branch_proj = nn.Linear(self.seq_len * self.d_model, self.d_model)
        self.feat_branch_proj = nn.Linear(self.seq_len * self.d_model, self.d_model)

        hidden_dim = self.d_model * 3 + node_emb_dim
        self.res_blocks = nn.ModuleList([_MLPRes(hidden_dim, dropout) for _ in range(3)])
        self.out_proj = nn.Linear(hidden_dim, self.pred_len * self.out_dim)

    def set_r(self, r: float) -> None:
        self._r = float(r)

    def _flatten_project(self, t: torch.Tensor, proj: nn.Linear, B: int, N: int, L: int) -> torch.Tensor:
        # t: [B, L, N, d_model] -> [B, N, L*d_model] -> proj -> [B, N, d_model]
        return proj(t.permute(0, 2, 1, 3).reshape(B, N, L * self.d_model))

    def forward(self, x: torch.Tensor, spat_adj: torch.Tensor, *, training: Optional[bool] = None):
        # x: [B, L, N, Dx]
        B, L, N, Dx = x.shape
        is_training = self.training if training is None else training

        h = self.start_fc(x)  # [B, L, N, d_model]

        # --- spatial branch ---
        spa_in = self.trans_spat(h.permute(0, 2, 1, 3).reshape(B, N, L * self.d_model))  # [B,N,d_model_spat]
        spa_out, att_spat, loss_spat = self.spat_gsat(spa_in, spat_adj, r=self._r, training=is_training)
        spa_out = self.inv_trans_spat(spa_out).view(B, N, L, self.d_model).permute(0, 2, 1, 3)  # [B,L,N,d_model]

        # --- temporal branch, applied to the spatial branch's output ---
        temp_in = self.trans_temp(spa_out.reshape(B, L, N * self.d_model))  # [B,L,d_model_temp]
        temp_out, att_temp, loss_temp = self.temp_gsat(temp_in, self.temp_adj, r=self._r, training=is_training)
        temp_out = self.inv_trans_temp(temp_out).view(B, L, N, self.d_model)  # [B,L,N,d_model]

        # --- feature branch (our extension), on the raw input (channels not yet mixed) ---
        feat_in = self.trans_feat(x.permute(0, 3, 1, 2).reshape(B, Dx, L * N))  # [B,Dx,d_model_feat]
        feat_out, att_feat, loss_feat = self.feat_gsat(feat_in, self.feat_adj, r=self._r, training=is_training)
        feat_out = self.inv_trans_feat(feat_out).view(B, Dx, L, N).permute(0, 2, 3, 1)  # [B,L,N,Dx]
        feat_out = self.start_fc(feat_out)  # [B,L,N,d_model] — fuse into the same embedding space

        # --- fuse branches + learned node identity ---
        res_emb = self._flatten_project(h, self.res_proj, B, N, L)
        temp_emb = self._flatten_project(temp_out, self.temp_branch_proj, B, N, L)
        feat_emb = self._flatten_project(feat_out, self.feat_branch_proj, B, N, L)
        node_emb = self.node_emb.unsqueeze(0).expand(B, -1, -1)  # [B,N,node_emb_dim]

        hidden = torch.cat([res_emb, temp_emb, feat_emb, node_emb], dim=-1)
        for block in self.res_blocks:
            hidden = block(hidden)

        out = self.out_proj(hidden).view(B, N, self.pred_len, self.out_dim).permute(0, 2, 1, 3)

        info_losses = {"spat": loss_spat, "temp": loss_temp, "feat": loss_feat}
        atts = {"spat": att_spat, "temp": att_temp, "feat": att_feat}
        return out, info_losses, atts


@register("stexplainer")
class STExplainerForecaster(GNNForecasterBase):
    """STExplainer forecaster: faithful GSAT-based port + a feature-axis extension.

    Exposes per-instance node/time/feature explanations via :meth:`explain`.
    """

    name: str = "stexplainer"
    d_model: int = 16
    d_model_spat: int = 64
    d_model_temp: int = 64
    d_model_feat: int = 32
    node_emb_dim: int = 32
    n_heads_l1: int = 4
    n_heads_l2: int = 1
    dropout: float = 0.15
    extractor_dropout: float = 0.5

    # sparsity-ratio anneal for the KL(Bernoulli(att) || Bernoulli(r)) info loss
    init_r: float = 0.9
    final_r: float = 0.5
    r_anneal_frac: float = 0.4

    # per-branch info-loss weights, ramped 0 -> target over [beta_ramp_start, beta_ramp_end]
    beta_spat: float = 1.0
    beta_temp: float = 1.0
    beta_feat: float = 1.0
    beta_ramp_start: float = 0.25
    beta_ramp_end: float = 0.5

    def _build_net(self, bundle, n_nodes, *, A_norm, device):
        A_dense = self._A_raw.to(device).to_dense()
        adj_mask = (A_dense > 0).float()
        eye = torch.eye(n_nodes, device=device)
        adj_mask = torch.clamp(adj_mask + eye, max=1.0)

        net = STExplainerNet(
            input_dim=len(bundle.x_cols),
            out_dim=len(bundle.y_cols),
            pred_len=int(bundle.raw.spec.pred_len),
            n_nodes=n_nodes,
            seq_len=int(bundle.raw.spec.seq_len),
            d_model=int(self.d_model),
            d_model_spat=int(self.d_model_spat),
            d_model_temp=int(self.d_model_temp),
            d_model_feat=int(self.d_model_feat),
            node_emb_dim=int(self.node_emb_dim),
            n_heads_l1=int(self.n_heads_l1),
            n_heads_l2=int(self.n_heads_l2),
            dropout=float(self.dropout),
            extractor_dropout=float(self.extractor_dropout),
        )
        net.register_buffer("spat_adj", adj_mask)
        return net

    def _r_schedule(self, progress: float) -> float:
        if progress >= self.r_anneal_frac:
            return float(self.final_r)
        frac = progress / max(self.r_anneal_frac, 1e-8)
        return float(self.init_r - (self.init_r - self.final_r) * frac)

    def _beta_schedule(self, progress: float, target: float) -> float:
        if progress < self.beta_ramp_start:
            return 0.0
        if progress < self.beta_ramp_end:
            span = max(self.beta_ramp_end - self.beta_ramp_start, 1e-8)
            return float(target) * (progress - self.beta_ramp_start) / span
        return float(target)

    def _graph_forward(self, net, x):
        net.set_r(self.final_r)
        pred, _, _ = net(x, net.spat_adj.to(x.device), training=False)
        return pred

    def _graph_forward_train(self, net, x, y_true, progress):
        net.set_r(self._r_schedule(progress))
        pred, info_losses, _ = net(x, net.spat_adj.to(x.device), training=True)
        aux = (
            self._beta_schedule(progress, self.beta_spat) * info_losses["spat"]
            + self._beta_schedule(progress, self.beta_temp) * info_losses["temp"]
            + self._beta_schedule(progress, self.beta_feat) * info_losses["feat"]
        )
        return pred, aux

    def explain(self, x: torch.Tensor, target_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Explanation for one already-built input ``x`` ([1, L, N, Dx]).

        Returns raw per-axis attention matrices (``att_spat [N,N]``, ``att_temp
        [L,L]``, ``att_feat [Dx,Dx]``, values in (0,1)) plus derived importances.

        Architectural note (from the real GSAT mechanism, not a simplification):
        only ``node_importance`` can be made **target-specific** — the spatial
        branch's graph nodes are literally the regions, and GAT attention rows
        are per-target by construction (``out[i] = sum_j alpha[i,j] * h[j]``), so
        passing ``target_idx`` returns "how much did each neighbor (row j / col j
        of ``att_spat[target_idx]``) contribute to *that* region's representation."
        ``time_importance``/``feature_importance`` are always **window-level**,
        shared across every region's forecast in this time window — the temporal
        and feature branches operate on representations already aggregated across
        all N nodes (temporal) or all L steps and N nodes (feature) *before*
        their GSAT pass runs, so there is no node-specific temporal/feature signal
        to extract even in principle. ``target_idx=None`` returns the whole-graph
        aggregate (mean over each node's incident edges) for all three axes.
        """
        if self._net is None:
            raise RuntimeError(f"{self.name} must be fit()/load_checkpoint()'d before explain()")
        dev = next(self._net.parameters()).device
        self._net.eval()
        with torch.no_grad():
            self._net.set_r(self.final_r)
            _, _, atts = self._net(x.to(dev), self._net.spat_adj.to(dev), training=False)

        def _aggregate(att: torch.Tensor, adj: torch.Tensor) -> np.ndarray:
            a = att[0].detach().cpu()  # first (only) batch item: [M,M]
            mask = adj.detach().cpu()
            deg = mask.sum(dim=1) + mask.sum(dim=0)
            score = (a.sum(dim=1) + a.sum(dim=0)) / deg.clamp_min(1.0)
            return score.numpy()

        def _target_node_importance(att_spat: torch.Tensor, idx: int) -> np.ndarray:
            a = att_spat[0].detach().cpu()  # [N,N]
            return ((a[idx, :] + a[:, idx]) / 2.0).numpy()

        att_spat, att_temp, att_feat = atts["spat"][0], atts["temp"][0], atts["feat"][0]
        node_importance = (
            _target_node_importance(atts["spat"], target_idx)
            if target_idx is not None
            else _aggregate(atts["spat"], self._net.spat_adj)
        )
        return {
            "att_spat": att_spat.detach().cpu().numpy(),
            "att_temp": att_temp.detach().cpu().numpy(),
            "att_feat": att_feat.detach().cpu().numpy(),
            "node_importance": node_importance,
            "time_importance": _aggregate(atts["temp"], self._net.temp_adj),
            "feature_importance": _aggregate(atts["feat"], self._net.feat_adj),
        }
