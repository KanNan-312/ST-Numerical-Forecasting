"""Unified GNN forecaster: GCN-TCN, GraphWaveNet, STGCN, STSGCN.

All four models share an identical training loop (tqdm bars, early stopping,
checkpoint save/load).  The only thing that differs per model is:
  1. How the PyTorch network is constructed  (_build_net)
  2. How the forward pass is called          (_graph_forward)

Dataloader note
---------------
GNN models use :class:`~st_numeric_baselines.data.graph_dataset.GraphWindowDataset`
which produces batches of shape ``[B, L, N, Dx]`` (all N nodes per item) rather
than the ``[B, L, Dx]`` batches produced by the standard DL dataloader where
each item is a single ZIP-code time window.  See
``st_numeric_baselines/data/graph_dataset.py`` for the full explanation.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from st_numeric_baselines.bundles.datatypes import ProcBundle
from st_numeric_baselines.data.graph_dataset import GraphWindowDataset, graph_collate
from st_numeric_baselines.graph.loader import load_graph
from st_numeric_baselines.graph.torch_adj import normalize_adj_sym, sparse_adj
from st_numeric_baselines.models.base import BaseForecaster
from st_numeric_baselines.models.registry import register

# Re-export the three existing PyTorch network modules so this file is the
# single GNN entry point — nothing outside this package needs to import them.
from st_numeric_baselines.models.gnn.gcn_tcn_geo import GeoGCN_TCN          # noqa: F401
from st_numeric_baselines.models.gnn.graph_wavenet import GraphWaveNet        # noqa: F401
from st_numeric_baselines.models.gnn.stgcn import STGCN                      # noqa: F401

try:
    from transformers import GPT2Config, GPT2Model
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


# ─────────────────────────────────────────────────────────────────────────────
# STSGCN  (Spatial-Temporal Synchronous GCN, AAAI 2020)
# ─────────────────────────────────────────────────────────────────────────────

def _build_st_adj(A_s_dense: torch.Tensor, T_local: int) -> torch.Tensor:
    """Construct the dense [T_local*N, T_local*N] spatial-temporal adjacency.

    Block structure::

        A_st = [ A_s  I    0  ]
               [ I    A_s  I  ]   (for T_local = 3)
               [ 0    I    A_s]

    Diagonal blocks carry spatial edges; off-diagonal identity blocks carry
    temporal connections between consecutive time steps.  Rows are normalised.
    """
    N = A_s_dense.shape[0]
    dev = A_s_dense.device
    TN = T_local * N

    A_st = torch.zeros(TN, TN, dtype=torch.float32, device=dev)

    for t in range(T_local):
        A_st[t * N : (t + 1) * N, t * N : (t + 1) * N] = A_s_dense

    I = torch.eye(N, dtype=torch.float32, device=dev)
    for t in range(T_local - 1):
        A_st[t * N : (t + 1) * N, (t + 1) * N : (t + 2) * N] = I
        A_st[(t + 1) * N : (t + 2) * N, t * N : (t + 1) * N] = I

    deg = A_st.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return A_st / deg


class _STSGCMBlock(nn.Module):
    """One Spatial-Temporal Synchronous Graph Convolutional Module.

    Applies two hops of GCN on the expanded spatial-temporal adjacency
    ``A_st ∈ R^{T_local·N × T_local·N}``, then returns the representation at
    the CENTRE time step: ``R^{B, N, C_out}``.
    """

    def __init__(self, c_in: int, c_out: int, T_local: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        self.T_local = int(T_local)
        self.fc1 = nn.Linear(c_in, c_out)
        self.fc2 = nn.Linear(c_out, c_out)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(c_out)

    def forward(self, x_local: torch.Tensor, A_st: torch.Tensor) -> torch.Tensor:
        # x_local: [B, T_local, N, C]
        B, T, N, C = x_local.shape
        TN = T * N
        h = x_local.reshape(B, TN, C)  # [B, TN, C]

        # Two-hop GCN on A_st
        h = torch.einsum("ij,bjc->bic", A_st, h)
        h = F.relu(self.fc1(h))
        h = torch.einsum("ij,bjc->bic", A_st, h)
        h = F.relu(self.fc2(h))
        h = self.drop(h)

        # Extract centre time step
        mid = (self.T_local // 2) * N
        return self.norm(h[:, mid : mid + N, :])  # [B, N, C_out]


class STSGCNet(nn.Module):
    """STSGCN network (AAAI 2020).

    For multi-step forecasting each output step gets its own STSGCM stack that
    processes the last ``T_local`` encoder steps and emits one prediction.

    Args:
        input_dim:  number of input features (Dx)
        hidden_dim: hidden channel width
        pred_len:   forecasting horizon H
        n_nodes:    number of graph nodes N
        T_local:    local temporal window size for the synchronous graph
        n_layers:   number of STSGCM layers per prediction branch
        dropout:    dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        pred_len: int,
        n_nodes: int,
        *,
        T_local: int = 3,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pred_len = int(pred_len)
        self.T_local = int(T_local)

        self.in_proj = nn.Linear(int(input_dim), int(hidden_dim))

        # Independent prediction branch per output step
        self.stsgcm_stacks = nn.ModuleList([
            nn.ModuleList([
                _STSGCMBlock(int(hidden_dim), int(hidden_dim), T_local=T_local, dropout=dropout)
                for _ in range(int(n_layers))
            ])
            for _ in range(int(pred_len))
        ])
        self.out_heads = nn.ModuleList([
            nn.Linear(int(hidden_dim), 1) for _ in range(int(pred_len))
        ])

    def forward(self, x: torch.Tensor, A_st: torch.Tensor) -> torch.Tensor:
        # x:    [B, L, N, F]
        # A_st: [T_local*N, T_local*N] dense
        B, L, N, _ = x.shape
        h = self.in_proj(x)  # [B, L, N, H]

        # Crop / pad to last T_local steps
        T = self.T_local
        if L >= T:
            h_local = h[:, -T:, :, :]          # [B, T_local, N, H]
        else:
            h_local = F.pad(h, (0, 0, 0, 0, T - L, 0))

        outs: List[torch.Tensor] = []
        for step_i in range(self.pred_len):
            h_i = h_local
            for block in self.stsgcm_stacks[step_i]:
                h_out = block(h_i, A_st)                       # [B, N, H]
                h_i = h_out.unsqueeze(1).expand(-1, T, -1, -1) # re-wrap for next layer
            outs.append(self.out_heads[step_i](h_out))         # [B, N, 1]

        return torch.stack(outs, dim=1)  # [B, pred_len, N, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Shared training base
# ─────────────────────────────────────────────────────────────────────────────

class GNNForecasterBase(BaseForecaster):
    """Common training / evaluation loop for all GNN forecasters.

    Subclasses implement :meth:`_build_net` to return the PyTorch network, and
    optionally override :meth:`_graph_forward` when the network signature
    differs (e.g. GraphWaveNet expects a *list* of supports).
    """

    # ── training hparams ──────────────────────────────────────────────────────
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    patience: int = 5
    batch_size: int = 8          # time windows per batch (not ZIP windows)
    max_train_batches: Optional[int] = None
    seed: int = 0

    # Whether this model needs a real adjacency at all. Every GNN model still
    # goes through GraphWindowDataset (one item = all N nodes x one time
    # window) regardless of this flag -- that's just the batch shape, not the
    # graph. Set False on subclasses whose _build_net/_graph_forward never
    # reference A_norm/A_raw (e.g. STAEformer's pure attention, ST-HHOL's
    # self-contained learned hypergraph) so they can run on a dataset with no
    # graph.npz / no dataset.graph.path at all.
    requires_graph: bool = True

    def __init__(self) -> None:
        self._net: Optional[nn.Module] = None
        self._A_norm: Optional[torch.Tensor] = None   # symmetric-normalized sparse adj, CPU
        self._A_raw: Optional[torch.Tensor] = None    # unnormalized sparse adj, CPU
        self._n_nodes: Optional[int] = None
        self._pred_len: Optional[int] = None
        self._graph_dataloaders: Optional[Dict[str, DataLoader]] = None
        self.train_history: List[Dict[str, Any]] = []
        # Current batch's time marks ([B, L, Dm], e.g. tod_frac/dow for
        # traffic data), refreshed right before every _graph_forward(_train)
        # call -- same instance-state pattern as self._A_norm/_A_raw, so
        # subclasses that want it (e.g. STAEformer/STID's tod/dow embeddings)
        # read it directly instead of it being a method parameter every one
        # of this class's ~15 existing _graph_forward overrides would
        # otherwise have to accept.
        self._x_mark: Optional[torch.Tensor] = None

    # ── subclass hooks ────────────────────────────────────────────────────────

    def _build_net(
        self,
        bundle: ProcBundle,
        n_nodes: int,
        *,
        A_norm: torch.Tensor,
        device: torch.device,
    ) -> nn.Module:
        raise NotImplementedError

    def _graph_forward(self, net: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Run network forward (eval/predict — always free-running). Default: ``net(x, A_norm)``."""
        return net(x, self._A_norm.to(x.device))

    def _graph_forward_train(
        self,
        net: nn.Module,
        x: torch.Tensor,
        y_true: torch.Tensor,
        progress: float,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Training-step forward pass. Default: identical to :meth:`_graph_forward`.

        Override for models that need the training-time ground truth (e.g.
        DCRNN's scheduled sampling) or an auxiliary loss term (e.g. a VQ/GIB
        regularizer). ``y_true`` is ``[B, pred_len, N, Dy]`` (same layout as
        the network's own output) and ``progress`` is ``global_step /
        total_steps`` across the whole training run, for decay schedules.

        Return either a plain ``y_hat`` tensor (identical to ``_graph_forward``),
        or a ``(y_hat, aux_loss)`` tuple — ``aux_loss`` is added to the main MSE
        loss before backprop. Never called outside training (eval/predict always
        use ``_graph_forward``, so no ground truth ever leaks into evaluation).
        """
        return self._graph_forward(net, x)

    def _make_optimizer(self, net: nn.Module) -> torch.optim.Optimizer:
        """Create the optimizer.  Override to filter trainable params for frozen backbones."""
        return Adam(net.parameters(), lr=float(self.lr), weight_decay=float(self.weight_decay))

    # ── checkpoint (override to also persist adjacency) ────────────────────────

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        if self._net is None:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "net": self._net,
            "A_norm": self._A_norm,
            "n_nodes": self._n_nodes,
            "pred_len": self._pred_len,
        }
        # STSGCN stores an additional dense A_st
        if hasattr(self, "_A_st"):
            payload["A_st"] = getattr(self, "_A_st")
        torch.save(payload, p)

    def load_checkpoint(self, path: Union[str, Path], *, device: Optional[torch.device] = None) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        ckpt = torch.load(p, map_location=device or "cpu", weights_only=False)
        self._net = ckpt["net"]
        self._A_norm = ckpt.get("A_norm")
        self._n_nodes = ckpt.get("n_nodes")
        self._pred_len = ckpt.get("pred_len")
        if "A_st" in ckpt:
            self._A_st = ckpt["A_st"]

    def setup_graph_dataloaders(self, bundle: ProcBundle) -> None:
        """Rebuild graph dataloaders without training (needed after load_checkpoint)."""
        dls: Dict[str, DataLoader] = {}
        for split in ("train", "val", "test"):
            ds = GraphWindowDataset(bundle, split=split)
            dls[split] = DataLoader(
                ds,
                batch_size=int(self.batch_size),
                shuffle=False,
                collate_fn=graph_collate,
                drop_last=False,
            )
        self._graph_dataloaders = dls

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, bundle: ProcBundle, *, device: Optional[torch.device] = None) -> None:
        dev = device if device is not None else torch.device("cpu")
        torch.manual_seed(int(self.seed))

        # Load the precomputed node adjacency graph (dataset-level config).
        raw_aligned = bundle.raw.aligned
        graph_cfg = bundle.raw.graph

        n_nodes = int(bundle.aligned_proc.values.shape[0])
        self._n_nodes = n_nodes
        self._pred_len = int(bundle.raw.spec.pred_len)

        if graph_cfg.path:
            geo = load_graph(graph_cfg.path, raw_aligned.zipcodes)
            A = sparse_adj(geo.edge_index[0], geo.edge_index[1], n_nodes,
                           weight=geo.edge_weight, device=dev)
            self._A_norm = normalize_adj_sym(A).cpu()
            self._A_raw = A.cpu()
        elif self.requires_graph:
            raise ValueError(
                f"{self.name} requires graph.path to be set in the dataset config, "
                "pointing at a graph.npz file with 'A' and 'ids' arrays"
            )
        # else: requires_graph=False and no graph.path -- self._A_norm/_A_raw
        # stay None (set in __init__); this model's _build_net/_graph_forward
        # never reference them.

        # Graph-structured dataloaders (one item = all N nodes × one time window)
        graph_dls: Dict[str, DataLoader] = {}
        for split in ("train", "val", "test"):
            ds = GraphWindowDataset(bundle, split=split)
            graph_dls[split] = DataLoader(
                ds,
                batch_size=int(self.batch_size),
                shuffle=(split == "train"),
                collate_fn=graph_collate,
                drop_last=False,
            )
        self._graph_dataloaders = graph_dls

        net = self._build_net(bundle, n_nodes, A_norm=self._A_norm, device=dev).to(dev)
        opt = self._make_optimizer(net)

        train_dl = graph_dls["train"]
        val_dl = graph_dls["val"]
        pred_len = self._pred_len

        best_val = math.inf
        best_state: Optional[Dict[str, torch.Tensor]] = None
        bad_epochs = 0
        self.train_history = []

        _max_bt = int(self.max_train_batches) if self.max_train_batches is not None else None
        _train_total = min(len(train_dl), _max_bt) if _max_bt is not None else len(train_dl)

        total_steps = max(1, int(self.epochs) * max(1, _train_total))
        global_step = 0

        epoch_bar = tqdm(range(int(self.epochs)), desc=f"[{self.name}]", unit="ep")
        for ep in epoch_bar:
            t_ep0 = time.perf_counter()

            # ── train ────────────────────────────────────────────────────────
            net.train()
            train_sse, train_n = 0.0, 0
            train_bar = tqdm(train_dl, desc="  train", total=_train_total, leave=False, unit="bt")
            for bi, batch in enumerate(train_bar):
                if _max_bt is not None and bi >= _max_bt:
                    train_bar.close()
                    break

                x = batch["x"].to(dev)       # [B, L, N, Dx]
                y_true = batch["y"].to(dev)   # [B*N, pred_len, Dy]
                self._x_mark = batch["x_mark"].to(dev) if "x_mark" in batch else None
                B = x.shape[0]

                # reshape y_true to the network's own output layout, [B, pred_len, N, Dy],
                # for models whose training-step forward needs the ground truth (e.g.
                # DCRNN's scheduled sampling) — inverse of the flatten below.
                y_true_graph = y_true.reshape(B, n_nodes, pred_len, y_true.shape[-1]).permute(0, 2, 1, 3)
                progress = global_step / total_steps
                out = self._graph_forward_train(net, x, y_true_graph, progress)
                global_step += 1
                if isinstance(out, tuple):
                    y_hat, aux_loss = out
                else:
                    y_hat, aux_loss = out, None

                Dy = y_hat.shape[-1]
                y_hat_flat = y_hat.permute(0, 2, 1, 3).reshape(B * n_nodes, pred_len, Dy)

                loss = F.mse_loss(y_hat_flat, y_true)
                if aux_loss is not None:
                    loss = loss + aux_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if float(self.grad_clip) > 0:
                    nn.utils.clip_grad_norm_(net.parameters(), max_norm=float(self.grad_clip))
                opt.step()

                train_sse += float(loss.detach().item()) * y_true.numel()
                train_n += y_true.numel()
                train_bar.set_postfix({"loss": f"{float(loss.detach().item()):.4g}"})

            # ── validate ─────────────────────────────────────────────────────
            net.eval()
            sse, n_val = 0.0, 0
            with torch.no_grad():
                val_bar = tqdm(val_dl, desc="  val  ", leave=False, unit="bt")
                for batch in val_bar:
                    x = batch["x"].to(dev)
                    y_true = batch["y"].to(dev)
                    self._x_mark = batch["x_mark"].to(dev) if "x_mark" in batch else None
                    B = x.shape[0]
                    y_hat = self._graph_forward(net, x)
                    Dy = y_hat.shape[-1]
                    y_hat_flat = y_hat.permute(0, 2, 1, 3).reshape(B * n_nodes, pred_len, Dy)
                    diff = (y_hat_flat - y_true).float()
                    sse += float((diff * diff).sum().item())
                    n_val += diff.numel()
                    val_bar.set_postfix({"mse": f"{sse / max(n_val, 1):.4g}"})

            val_mse = sse / max(n_val, 1)
            train_mse = train_sse / max(train_n, 1)
            ep_time = time.perf_counter() - t_ep0

            self.train_history.append({
                "epoch": int(ep + 1),
                "train_mse": float(train_mse),
                "val_mse": float(val_mse),
                "epoch_time_sec": float(ep_time),
            })

            if val_mse < best_val - 1e-12:
                best_val = val_mse
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(self.patience):
                    epoch_bar.close()
                    break

            epoch_bar.set_postfix({
                "train": f"{train_mse:.4g}",
                "val":   f"{val_mse:.4g}",
                "best":  f"{best_val:.4g}",
            })

        if best_state is not None:
            net.load_state_dict(best_state)
        self._net = net

    # ── predict_batch ─────────────────────────────────────────────────────────

    def predict_batch(
        self,
        batch: Dict[str, Any],
        *,
        bundle: ProcBundle,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError(f"{self.name} must be fit() before predict_batch()")
        dev = device if device is not None else next(self._net.parameters()).device
        self._net.to(dev)
        self._net.eval()

        x = batch["x"].to(dev)  # [B, L, N, Dx]
        self._x_mark = batch["x_mark"].to(dev) if "x_mark" in batch else None
        B, L, N, _ = x.shape

        with torch.no_grad():
            y_hat = self._graph_forward(self._net, x)  # [B, pred_len, N, Dy]

        Dy = y_hat.shape[-1]
        return y_hat.permute(0, 2, 1, 3).reshape(B * N, self._pred_len, Dy)


# ─────────────────────────────────────────────────────────────────────────────
# Concrete forecasters
# ─────────────────────────────────────────────────────────────────────────────

@register("gcn_tcn")
class GCNTCNForecaster(GNNForecasterBase):
    """GCN + TCN forecaster on a geographic k-NN graph."""

    name: str = "gcn_tcn"
    hidden_dim: int = 32
    dropout: float = 0.1
    tcn_kernel: int = 3

    def _build_net(self, bundle, n_nodes, *, A_norm, device):
        return GeoGCN_TCN(
            input_dim=len(bundle.x_cols),
            hidden_dim=int(self.hidden_dim),
            pred_len=int(bundle.raw.spec.pred_len),
            dropout=float(self.dropout),
            tcn_kernel=int(self.tcn_kernel),
        )

    def _graph_forward(self, net, x):
        return net(x, self._A_norm.to(x.device))


@register("graph_wavenet")
class GraphWaveNetForecaster(GNNForecasterBase):
    """Graph WaveNet forecaster with learnable adaptive adjacency."""

    name: str = "graph_wavenet"
    residual_channels: int = 32
    dilation_channels: int = 32
    skip_channels: int = 64
    end_channels: int = 128
    kernel_size: int = 2
    n_blocks: int = 2
    n_layers: int = 2
    gcn_order: int = 2
    dropout: float = 0.1
    adaptive_adj: bool = True
    adaptive_emb_dim: int = 8

    def __init__(self) -> None:
        super().__init__()
        self._base_edge_index: Optional[torch.Tensor] = None

    def _build_net(self, bundle, n_nodes, *, A_norm, device):
        # Convert sparse A_norm to COO edge_index for the adaptive adj module
        A_dense = A_norm.to(device).to_dense()
        r, c = A_dense.nonzero(as_tuple=True)
        self._base_edge_index = torch.stack([r.cpu(), c.cpu()], dim=0)

        return GraphWaveNet(
            input_dim=len(bundle.x_cols),
            pred_len=int(bundle.raw.spec.pred_len),
            n_nodes=n_nodes,
            residual_channels=int(self.residual_channels),
            dilation_channels=int(self.dilation_channels),
            skip_channels=int(self.skip_channels),
            end_channels=int(self.end_channels),
            kernel_size=int(self.kernel_size),
            n_blocks=int(self.n_blocks),
            n_layers=int(self.n_layers),
            gcn_order=int(self.gcn_order),
            dropout=float(self.dropout),
            adaptive_adj=bool(self.adaptive_adj),
            adaptive_emb_dim=int(self.adaptive_emb_dim),
            base_edge_index=self._base_edge_index,
        )

    def _graph_forward(self, net, x):
        # GraphWaveNet expects a list of support matrices
        return net(x, [self._A_norm.to(x.device)])


@register("stgcn")
class STGCNForecaster(GNNForecasterBase):
    """STGCN: Chebyshev graph conv + gated temporal conv blocks."""

    name: str = "stgcn"
    hidden_dim: int = 32
    n_blocks: int = 2
    Kt: int = 3
    dropout: float = 0.1

    def _build_net(self, bundle, n_nodes, *, A_norm, device):
        return STGCN(
            input_dim=len(bundle.x_cols),
            hidden_dim=int(self.hidden_dim),
            pred_len=int(bundle.raw.spec.pred_len),
            n_blocks=int(self.n_blocks),
            Kt=int(self.Kt),
            dropout=float(self.dropout),
        )

    def _graph_forward(self, net, x):
        return net(x, self._A_norm.to(x.device))


@register("stsgcn")
class STSGCNForecaster(GNNForecasterBase):
    """STSGCN: Spatial-Temporal Synchronous GCN (Wu et al., AAAI 2020).

    Builds a spatial-temporal synchronous adjacency by stacking T_local copies
    of the spatial graph (diagonal blocks) and connecting consecutive time steps
    with identity matrices (off-diagonal blocks).  A GCN applied to this
    T_local·N × T_local·N graph captures spatial and temporal correlations
    **synchronously** in a single operation, without the sequential
    spatial→temporal decomposition used by GCN-TCN or STGCN.

    Reference: https://github.com/Davidham3/STSGCN
    """

    name: str = "stsgcn"
    hidden_dim: int = 32
    T_local: int = 3
    n_layers: int = 2
    dropout: float = 0.1

    def __init__(self) -> None:
        super().__init__()
        self._A_st: Optional[torch.Tensor] = None  # dense [T_local*N, T_local*N], CPU

    def _build_net(self, bundle, n_nodes, *, A_norm, device):
        A_dense = A_norm.to(device).to_dense()
        self._A_st = _build_st_adj(A_dense, int(self.T_local)).cpu()

        return STSGCNet(
            input_dim=len(bundle.x_cols),
            hidden_dim=int(self.hidden_dim),
            pred_len=int(bundle.raw.spec.pred_len),
            n_nodes=n_nodes,
            T_local=int(self.T_local),
            n_layers=int(self.n_layers),
            dropout=float(self.dropout),
        )

    def _graph_forward(self, net, x):
        return net(x, self._A_st.to(x.device))


# ─────────────────────────────────────────────────────────────────────────────
# ST-LLM+  (Partially Frozen Graph Attention + GPT-2 backbone, 2024)
# ─────────────────────────────────────────────────────────────────────────────

class _SpatialGraphConv(nn.Module):
    """Single-layer GCN that aggregates node features over the geographic graph.

    Applied independently at each patch-token position so spatial message
    passing and temporal attention remain on the same time scale.
    """

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.W = nn.Linear(d_model, d_model, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # h: [B, T_p, N, D]   A: [N, N] dense
        # h_agg[b,t,n,d] = sum_m A[n,m] * h[b,t,m,d]
        h_agg = torch.einsum("nm,btmd->btnd", A, h)
        h_agg = self.drop(self.act(self.W(h_agg)))
        return self.norm(h + h_agg)   # residual + layer norm


class _STLLMPlusNet(nn.Module):
    """ST-LLM+ network.

    Architecture (per GPT-2 transformer block):
    1. Temporal: standard GPT-2 self-attention (frozen except LayerNorm weights)
    2. Spatial:  single-layer GCN over the geographic k-NN graph (fully trained)
    3. Fusion:   ``h = h_temporal + sigmoid(gate) * h_spatial``

    This is the Partially Frozen Graph Attention (PFGA) mechanism from the
    ST-LLM+ paper, adapted to use GPT-2 as the backbone (original uses GPT-2
    as well, making this a faithful re-implementation rather than an approximation).

    Input/output shapes follow :class:`GNNForecasterBase` convention:
      forward(x, A_norm) → [B, pred_len, N, out_dim]
    """

    def __init__(
        self,
        *,
        seq_len: int,
        pred_len: int,
        input_dim: int,
        out_dim: int,
        patch_len: int,
        stride: int,
        gpt_layers: int,
        dropout: float,
        pretrained: bool,
    ) -> None:
        super().__init__()
        if not _HAS_TRANSFORMERS:
            raise ImportError(
                "STLLMPlusForecaster requires the `transformers` package.\n"
                "Install it with:  pip install transformers"
            )

        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.out_dim = int(out_dim)
        self.patch_len = int(patch_len)
        self.stride = int(stride)

        patch_num = (int(seq_len) - int(patch_len)) // int(stride) + 1
        self.patch_num = patch_num
        D_LLM = 768  # GPT-2 hidden dimension

        # Project flattened patch (patch_len × input_dim) into GPT-2 token space
        self.in_layer = nn.Linear(int(patch_len) * int(input_dim), D_LLM)

        # GPT-2 backbone
        if pretrained:
            self.gpt2 = GPT2Model.from_pretrained("gpt2")
        else:
            cfg = GPT2Config(n_embd=D_LLM, n_layer=max(1, int(gpt_layers)), n_head=12)
            self.gpt2 = GPT2Model(cfg)

        # Use only the first gpt_layers transformer blocks
        self.gpt2.h = self.gpt2.h[: int(gpt_layers)]

        # Freeze backbone — only LayerNorm weights are adapted (same as GPT4TS)
        for param in self.gpt2.parameters():
            param.requires_grad = False
        for name, param in self.gpt2.named_parameters():
            if "ln_" in name:
                param.requires_grad = True

        # PFGA: one spatial GCN + one learnable fusion gate per GPT-2 block
        n_blocks = int(gpt_layers)
        self.graph_convs = nn.ModuleList([
            _SpatialGraphConv(D_LLM, float(dropout)) for _ in range(n_blocks)
        ])
        self.fusion_gates = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(n_blocks)
        ])

        self.drop = nn.Dropout(float(dropout))
        # Flatten all patch token hidden states → pred_len × out_dim per node
        self.out_layer = nn.Linear(D_LLM * patch_num, int(pred_len) * int(out_dim))

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # x:      [B, L, N, Dx]
        # A_norm: [N, N] sparse or dense (k-NN geographic adjacency)
        B, L, N, Dx = x.shape
        if L > self.seq_len:
            x = x[:, -self.seq_len :, :, :]

        # Per-node instance normalisation (handles inter-ZIP distribution shift)
        means = x.mean(dim=1, keepdim=True).detach()          # [B, 1, N, Dx]
        x = x - means
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / std

        # Materialise dense adjacency for einsum (N is small, ~20–500 nodes)
        A = A_norm.to(x.device)
        if A.is_sparse:
            A = A.to_dense()

        # Flatten nodes into batch axis for patch processing: [B*N, L, Dx]
        x_bn = x.permute(0, 2, 1, 3).reshape(B * N, L, Dx)

        # Extract patches: [B*N, Np, Dx, patch_len] → flatten → [B*N, Np, Dx*patch_len]
        xp = x_bn.unfold(1, self.patch_len, self.stride)
        Np = xp.shape[1]
        xp = xp.reshape(B * N, Np, Dx * self.patch_len)

        # Embed patches + add GPT-2 positional encoding
        h = self.in_layer(xp)                                  # [B*N, Np, D_LLM]
        pos_ids = torch.arange(Np, device=h.device)
        h = h + self.gpt2.wpe(pos_ids).unsqueeze(0)            # broadcast over B*N
        h = self.drop(h)

        # ── PFGA: interleave temporal (GPT-2 block) + spatial (graph conv) ──
        for block, graph_conv, gate in zip(
            self.gpt2.h, self.graph_convs, self.fusion_gates
        ):
            # Temporal attention (frozen weights, trained LayerNorms)
            h_t = block(h)[0]                                  # [B*N, Np, D_LLM]

            # Spatial: bring N back as a dimension, apply GCN, flatten again
            #   [B*N, Np, D] → [B, N, Np, D] → [B, Np, N, D] → GCN → [B, Np, N, D]
            #                → [B, N, Np, D] → [B*N, Np, D]
            h_4d = h_t.reshape(B, N, Np, -1).permute(0, 2, 1, 3)   # [B, Np, N, D]
            h_sp = graph_conv(h_4d, A)                               # [B, Np, N, D]
            h_sp = h_sp.permute(0, 2, 1, 3).reshape(B * N, Np, -1)  # [B*N, Np, D]

            # Gated fusion: gate initialised at 0 so training starts from temporal-only
            h = h_t + torch.sigmoid(gate) * h_sp

        # ── Output ──────────────────────────────────────────────────────────
        out = self.out_layer(h.reshape(B * N, -1))             # [B*N, pred_len * out_dim]
        out = out.view(B, N, self.pred_len, self.out_dim)
        out = out.permute(0, 2, 1, 3)                          # [B, pred_len, N, out_dim]

        # Denormalise: broadcast [B, 1, N, out_dim] over [B, pred_len, N, out_dim]
        out = out * std[:, :, :, : self.out_dim] + means[:, :, :, : self.out_dim]
        return out


@register("stllm_plus")
class STLLMPlusForecaster(GNNForecasterBase):
    """ST-LLM+: Partially Frozen Graph Attention interleaved with a frozen GPT-2 backbone.

    At each GPT-2 transformer block, a trainable graph convolutional layer
    aggregates neighbourhood information from the geographic k-NN graph, and
    the result is fused with the temporal attention output via a learnable gate.
    Only GPT-2 LayerNorm weights and the spatial GCN layers are trained.

    Reference: KanNan-312/ST-LLM-Plus (2024) — faithful re-implementation using
    GPT-2 (which the original paper also supports alongside LLaMA variants).

    Requires the ``transformers`` library (``pip install transformers``).
    GPT-2 weights (~500 MB) are downloaded automatically from HuggingFace.
    """

    name: str = "stllm_plus"

    # model hparams
    patch_len: int = 3
    stride: int = 1
    gpt_layers: int = 6
    dropout: float = 0.1
    pretrained: bool = True

    # training hparams (smaller batch: GPT-2 forward is memory-heavy)
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    patience: int = 3
    batch_size: int = 4
    max_train_batches: Optional[int] = None
    seed: int = 0

    def fit(self, bundle: ProcBundle, *, device: Optional[torch.device] = None) -> None:
        if not _HAS_TRANSFORMERS:
            raise ImportError(
                "STLLMPlusForecaster requires the `transformers` package.\n"
                "Install it with:  pip install transformers"
            )
        super().fit(bundle, device=device)

    def _build_net(
        self,
        bundle: ProcBundle,
        n_nodes: int,
        *,
        A_norm: torch.Tensor,
        device: torch.device,
    ) -> nn.Module:
        seq_len = int(bundle.raw.spec.seq_len)
        patch_len = int(self.patch_len)
        stride = int(self.stride)
        if patch_len > seq_len:
            patch_len = seq_len
            stride = seq_len

        return _STLLMPlusNet(
            seq_len=seq_len,
            pred_len=int(bundle.raw.spec.pred_len),
            input_dim=int(len(bundle.x_cols)),
            out_dim=int(len(bundle.y_cols)),
            patch_len=patch_len,
            stride=stride,
            gpt_layers=int(self.gpt_layers),
            dropout=float(self.dropout),
            pretrained=bool(self.pretrained),
        )

    def _graph_forward(self, net: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return net(x, self._A_norm.to(x.device))

    def _make_optimizer(self, net: nn.Module) -> torch.optim.Optimizer:
        # Only optimise non-frozen parameters: LayerNorms + graph convs + output head
        trainable = [p for p in net.parameters() if p.requires_grad]
        return Adam(trainable, lr=float(self.lr), weight_decay=float(self.weight_decay))
