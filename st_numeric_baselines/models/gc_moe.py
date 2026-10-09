"""GC-MoE — "Graph-Conditioned Mixture of Graph Neural Network Experts for
Traffic Forecasting" (Ghaffari, Sheikhi & Gilman).

Reference: https://github.com/Ahghaffari/gc_moe  (arXiv:2605.30486)

Confirmed from source (``src/moe_model.py``'s ``GraphConditionedExpertRouter``
/ ``MoE_STLoRA``): experts are **frozen, independently pretrained** GNN
models — this registry trains them via each sub-model's own ``fit()``
first (reusing ``ensemble_st.py``'s delegation pattern), then freezes every
parameter. A small ("~17K param", per the paper) **router** combines a
*static* pathway (9 real graph-topology features — degree, closeness,
clustering, PageRank, betweenness, k-core, eigenvector centrality, the
Fiedler vector, and the 3rd Laplacian eigenvector — computed once from the
real adjacency) with a *dynamic* pathway (a temporal-attention summary of
the current input window), fused through a sigmoid gate, to produce
per-node **soft** mixture weights over the frozen experts (confirmed: a
weighted sum, not hard top-1 routing like this registry's ``testam.py``).
Only the router is trained — ``requires_grad_(False)`` on every expert.

``requires_graph = True`` at the module level here (not a
``GNNForecasterBase`` subclass, so there's no class attribute for it, but
``fit()`` raises the same way if ``graph.path`` isn't set) — unlike
``ensemble_st``/``testam``, GC-MoE's whole premise is a router conditioned
on *real* graph topology, so a real adjacency is required, not optional.

**Caveat, stated plainly**: the layer-by-layer router architecture below
(``GraphConditionedRouter``) was confirmed from a direct fetch of
``src/moe_model.py``'s actual class definitions. How the *router's two
logit terms* (`static_routing` — a raw per-node learnable bias — and
`dynamic_logits` — produced from the gate-fused static/dynamic
representation) are summed, and the "noisy top-k" gating detail, are
reconstructed from the fetch's own description rather than a verified
line-by-line copy — see ``GraphConditionedRouter.forward``.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from st_numeric_baselines.bundles.datatypes import ProcBundle
from st_numeric_baselines.graph.loader import load_graph
from st_numeric_baselines.graph.torch_adj import sparse_adj
from st_numeric_baselines.models.base import BaseForecaster
from st_numeric_baselines.models.hparams import apply_hparams
from st_numeric_baselines.models.registry import get as get_model
from st_numeric_baselines.models.registry import register

try:
    import networkx as nx
    _HAS_NETWORKX = True
except Exception:  # pragma: no cover
    nx = None
    _HAS_NETWORKX = False

_N_GRAPH_FEATS = 9


def _compute_graph_features(A: np.ndarray) -> np.ndarray:
    """9 static per-node graph-topology features from a real adjacency
    (degree, closeness, clustering, PageRank, betweenness, k-core,
    eigenvector centrality, Fiedler vector, 3rd Laplacian eigenvector)."""
    if not _HAS_NETWORKX:
        raise ImportError("gc_moe requires the `networkx` package for its graph-topology router features.\n"
                           "Install it with:  pip install networkx")
    n = A.shape[0]
    G = nx.from_numpy_array(np.asarray(A))

    degree = dict(G.degree())
    closeness = nx.closeness_centrality(G)
    clustering = nx.clustering(G)
    try:
        pagerank = nx.pagerank(G)
    except Exception:
        pagerank = {i: 1.0 / max(n, 1) for i in range(n)}
    try:
        betweenness = nx.betweenness_centrality(G)
    except Exception:
        betweenness = {i: 0.0 for i in range(n)}
    kcore = nx.core_number(nx.Graph(G))  # core_number requires a simple graph (no self-loops/multi-edges)
    try:
        eigenvector = nx.eigenvector_centrality_numpy(G)
    except Exception:
        eigenvector = {i: 0.0 for i in range(n)}

    L = nx.laplacian_matrix(G).toarray().astype(np.float64)
    eigvals, eigvecs = np.linalg.eigh(L)
    fiedler = eigvecs[:, 1] if n > 1 else np.zeros(n)
    third = eigvecs[:, 2] if n > 2 else np.zeros(n)

    feats = np.zeros((n, _N_GRAPH_FEATS), dtype=np.float32)
    for i in range(n):
        feats[i] = [
            degree.get(i, 0), closeness.get(i, 0.0), clustering.get(i, 0.0), pagerank.get(i, 0.0),
            betweenness.get(i, 0.0), kcore.get(i, 0), eigenvector.get(i, 0.0), fiedler[i], third[i],
        ]
    return feats


class GraphConditionedRouter(nn.Module):
    """Static (graph-topology) + dynamic (input-window) dual-pathway router
    producing per-node soft mixture weights over ``n_experts`` frozen models."""

    def __init__(
        self,
        n_nodes: int,
        n_experts: int,
        input_dim: int,
        *,
        embed_dim: int = 32,
        dynamic_dim: int = 32,
        top_k: Optional[int] = None,
        temperature: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_experts = int(n_experts)
        self.top_k = int(top_k) if top_k is not None else self.n_experts
        self.temperature = float(temperature)

        # static pathway
        self.feature_projection = nn.Sequential(
            nn.Linear(_N_GRAPH_FEATS, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.node_embedding = nn.Parameter(torch.randn(n_nodes, embed_dim) * 0.01)
        self.static_routing = nn.Parameter(torch.randn(n_nodes, self.n_experts) * 0.01)
        self.register_buffer("graph_feats", torch.zeros(n_nodes, _N_GRAPH_FEATS), persistent=False)

        # dynamic pathway
        self.input_encoder = nn.Sequential(nn.Linear(input_dim, dynamic_dim), nn.ReLU(), nn.Dropout(dropout))
        self.temporal_attn = nn.Linear(dynamic_dim, 1)
        self.dynamic_projection = nn.Sequential(nn.Linear(dynamic_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout))

        # fusion + output
        self.adaptive_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1), nn.Sigmoid(),
        )
        self.dynamic_router_head = nn.Sequential(
            nn.Linear(embed_dim, self.n_experts), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(self.n_experts, self.n_experts),
        )

    def set_graph_features(self, feats: torch.Tensor) -> None:
        self.graph_feats = feats.to(self.graph_feats.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, N, Dx]
        B, L, N, _ = x.shape

        static_repr = self.node_embedding + self.feature_projection(self.graph_feats)  # [N, embed_dim]
        static_exp = static_repr.unsqueeze(0).expand(B, -1, -1)  # [B, N, embed_dim]

        h = self.input_encoder(x)  # [B, L, N, dynamic_dim]
        attn_weights = F.softmax(self.temporal_attn(h), dim=1)  # [B, L, N, 1]
        dyn = (h * attn_weights).sum(dim=1)  # [B, N, dynamic_dim]
        dyn_repr = self.dynamic_projection(dyn)  # [B, N, embed_dim]

        gate = self.adaptive_gate(torch.cat([static_exp, dyn_repr], dim=-1))  # [B, N, 1]
        fused = gate * static_exp + (1.0 - gate) * dyn_repr  # [B, N, embed_dim]

        dynamic_logits = self.dynamic_router_head(fused)  # [B, N, n_experts]
        logits = self.static_routing.unsqueeze(0) + dynamic_logits  # [B, N, n_experts]

        if self.training:
            logits = logits + 0.1 * torch.randn_like(logits)  # noisy top-k gating (Shazeer et al.), simplified

        if self.top_k < self.n_experts:
            topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
            masked = torch.full_like(logits, float("-inf"))
            masked.scatter_(-1, topk_idx, topk_vals)
            logits = masked

        return F.softmax(logits / self.temperature, dim=-1)  # [B, N, n_experts]


@register("gc_moe")
class GCMoEForecaster(BaseForecaster):
    """GC-MoE: frozen pretrained GNN experts + a small trainable graph-conditioned router."""

    name: str = "gc_moe"

    def __init__(self) -> None:
        self.expert_models: List[str] = ["gcn_tcn", "stgformer", "dcrnn"]
        self.expert_hparams: Dict[str, Dict[str, Any]] = {}
        self.embed_dim: int = 32
        self.dynamic_dim: int = 32
        self.top_k: Optional[int] = None  # None -> dense softmax over all experts
        self.temperature: float = 1.0
        self.router_epochs: int = 20
        self.router_lr: float = 1e-3
        self.router_patience: int = 5

        self._experts: List[BaseForecaster] = []
        self._expert_names: List[str] = []
        self._router: Optional[GraphConditionedRouter] = None
        self._n_nodes: Optional[int] = None
        self._pred_len: Optional[int] = None
        self._graph_dataloaders = None  # duck-typed "graph mode" signal, see ensemble_st.py

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, bundle: ProcBundle, *, device: Optional[torch.device] = None) -> None:
        if not self.expert_models:
            raise ValueError(f"{self.name}.expert_models must be a non-empty list of registered model names")
        graph_cfg = bundle.raw.graph
        if not graph_cfg.path:
            raise ValueError(
                f"{self.name} requires graph.path to be set (its router is conditioned on real graph "
                "topology features) -- unlike ensemble_st, this is not optional"
            )
        dev = device if device is not None else torch.device("cpu")

        # stage 1: train each expert normally, then freeze every parameter
        self._expert_names = list(self.expert_models)
        self._experts = []
        for mname in self._expert_names:
            sub = get_model(mname)
            apply_hparams(sub, (self.expert_hparams or {}).get(mname, {}))
            sub.fit(bundle, device=dev)
            net = getattr(sub, "_net", None)
            if net is not None:
                for p in net.parameters():
                    p.requires_grad_(False)
            self._experts.append(sub)

        self._graph_dataloaders = None
        for sub in self._experts:
            if getattr(sub, "_graph_dataloaders", None) is not None:
                self._graph_dataloaders = sub._graph_dataloaders
                break
        if self._graph_dataloaders is None:
            raise ValueError(f"{self.name} requires every expert_models entry to be a GNN model")

        n_nodes = int(bundle.aligned_proc.values.shape[0])
        self._n_nodes = n_nodes
        self._pred_len = int(bundle.raw.spec.pred_len)

        # stage 2: router, conditioned on the real adjacency's topology features
        geo = load_graph(graph_cfg.path, bundle.raw.aligned.zipcodes)
        A_dense = sparse_adj(geo.edge_index[0], geo.edge_index[1], n_nodes, weight=geo.edge_weight, device=dev).to_dense()
        graph_feats = _compute_graph_features(A_dense.cpu().numpy())

        router = GraphConditionedRouter(
            n_nodes, len(self._experts), len(bundle.x_cols),
            embed_dim=int(self.embed_dim), dynamic_dim=int(self.dynamic_dim),
            top_k=self.top_k, temperature=float(self.temperature),
        ).to(dev)
        router.set_graph_features(torch.tensor(graph_feats, dtype=torch.float32, device=dev))

        opt = Adam(router.parameters(), lr=float(self.router_lr))
        train_dl, val_dl = self._graph_dataloaders["train"], self._graph_dataloaders["val"]

        best_val = math.inf
        best_state = None
        bad_epochs = 0
        epoch_bar = tqdm(range(int(self.router_epochs)), desc=f"[{self.name}]", unit="ep")
        for _ep in epoch_bar:
            router.train()
            train_sse, train_n = 0.0, 0
            for batch in train_dl:
                loss, combined_flat, y_true = self._router_step(router, batch, bundle=bundle, device=dev)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                train_sse += float(loss.detach().item()) * y_true.numel()
                train_n += y_true.numel()

            router.eval()
            val_sse, val_n = 0.0, 0
            with torch.no_grad():
                for batch in val_dl:
                    _loss, combined_flat, y_true = self._router_step(router, batch, bundle=bundle, device=dev)
                    diff = (combined_flat - y_true).float()
                    val_sse += float((diff * diff).sum().item())
                    val_n += diff.numel()

            val_mse = val_sse / max(val_n, 1)
            epoch_bar.set_postfix({"train_mse": f"{train_sse / max(train_n,1):.4g}", "val_mse": f"{val_mse:.4g}"})

            if val_mse < best_val - 1e-12:
                best_val = val_mse
                best_state = {k: v.detach().cpu().clone() for k, v in router.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(self.router_patience):
                    epoch_bar.close()
                    break

        if best_state is not None:
            router.load_state_dict(best_state)
        self._router = router

    def _router_step(self, router, batch, *, bundle, device):
        x = batch["x"].to(device)
        y_true = batch["y"].to(device)  # [B*N, pred_len, Dy]
        B, _L, N, _Dx = x.shape

        weights = router(x)  # [B, N, n_experts]
        expert_outs = []
        for sub in self._experts:
            pred = sub.predict_batch(batch, bundle=bundle, device=device)  # [B*N, pred_len, Dy], no grad
            Dy = pred.shape[-1]
            expert_outs.append(pred.reshape(B, N, self._pred_len, Dy).permute(0, 2, 1, 3))  # [B,pred_len,N,Dy]
        stacked = torch.stack(expert_outs, dim=0)  # [E, B, pred_len, N, Dy]

        E = len(self._experts)
        Dy = stacked.shape[-1]
        w = weights.permute(2, 0, 1).reshape(E, B, 1, N, 1)  # [E, B, 1, N, 1]
        combined = (stacked * w).sum(dim=0)  # [B, pred_len, N, Dy]
        combined_flat = combined.permute(0, 2, 1, 3).reshape(B * N, self._pred_len, Dy)

        loss = F.mse_loss(combined_flat, y_true)
        return loss, combined_flat, y_true

    # ── predict ──────────────────────────────────────────────────────────────

    def predict_batch(
        self,
        batch: Dict[str, Any],
        *,
        bundle: ProcBundle,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if self._router is None:
            raise RuntimeError(f"{self.name} must be fit() before predict_batch()")
        dev = device if device is not None else next(self._router.parameters()).device
        x = batch["x"].to(dev)
        B, _L, N, _Dx = x.shape

        self._router.eval()
        with torch.no_grad():
            weights = self._router(x)  # [B, N, n_experts]
            expert_outs = []
            for sub in self._experts:
                pred = sub.predict_batch(batch, bundle=bundle, device=dev)  # [B*N, pred_len, Dy]
                Dy = pred.shape[-1]
                expert_outs.append(pred.reshape(B, N, self._pred_len, Dy).permute(0, 2, 1, 3))
            stacked = torch.stack(expert_outs, dim=0)  # [E, B, pred_len, N, Dy]
            E = len(self._experts)
            Dy = stacked.shape[-1]
            w = weights.permute(2, 0, 1).reshape(E, B, 1, N, 1)
            combined = (stacked * w).sum(dim=0)  # [B, pred_len, N, Dy]

        return combined.permute(0, 2, 1, 3).reshape(B * N, self._pred_len, Dy)

    # ── checkpoint ───────────────────────────────────────────────────────────

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        if self._router is None:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "expert_names": self._expert_names,
                "router_state": self._router.state_dict(),
                "router_init": {
                    "n_nodes": self._router.node_embedding.shape[0],
                    "n_experts": self._router.n_experts,
                    "input_dim": self._router.input_encoder[0].in_features,
                    "embed_dim": int(self.embed_dim),
                    "dynamic_dim": int(self.dynamic_dim),
                    "top_k": self._router.top_k,
                    "temperature": self._router.temperature,
                },
                "graph_feats": self._router.graph_feats.cpu(),
                "n_nodes": self._n_nodes,
                "pred_len": self._pred_len,
            },
            p,
        )
        for mname, sub in zip(self._expert_names, self._experts):
            sub.save_checkpoint(p.parent / f"{p.stem}__{mname}.pt")

    def load_checkpoint(self, path: Union[str, Path], *, device: Optional[torch.device] = None) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        dev = device or torch.device("cpu")
        payload = torch.load(p, map_location=dev, weights_only=False)

        self._expert_names = list(payload["expert_names"])
        self._experts = []
        for mname in self._expert_names:
            sub = get_model(mname)
            apply_hparams(sub, (self.expert_hparams or {}).get(mname, {}))
            sub.load_checkpoint(p.parent / f"{p.stem}__{mname}.pt", device=dev)
            self._experts.append(sub)

        init = payload["router_init"]
        router = GraphConditionedRouter(
            init["n_nodes"], init["n_experts"], init["input_dim"],
            embed_dim=init["embed_dim"], dynamic_dim=init["dynamic_dim"],
            top_k=init["top_k"], temperature=init["temperature"],
        ).to(dev)
        router.load_state_dict(payload["router_state"])
        router.set_graph_features(payload["graph_feats"].to(dev))
        self._router = router
        self._n_nodes = payload["n_nodes"]
        self._pred_len = payload["pred_len"]

    def setup_graph_dataloaders(self, bundle: ProcBundle) -> None:
        """Rebuild every expert's graph dataloaders after load_checkpoint()
        (mirrors GNNForecasterBase.setup_graph_dataloaders)."""
        for sub in self._experts:
            if hasattr(sub, "setup_graph_dataloaders"):
                sub.setup_graph_dataloaders(bundle)
                if self._graph_dataloaders is None and getattr(sub, "_graph_dataloaders", None) is not None:
                    self._graph_dataloaders = sub._graph_dataloaders
