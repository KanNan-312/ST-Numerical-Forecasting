"""CaST — Causal Spatio-Temporal representation learning (yutong-xia/CaST).

Reference: https://github.com/yutong-xia/CaST  (arXiv:2309.13378)

CaST does not require externally labeled "environments" — it discovers
pseudo-environments from the training data itself via a small VQ codebook, so
it works directly on this benchmark's single chronologically-split panel (no
pipeline changes needed, per the research pass on this model).

Two adaptations from the paper, both computed once in ``_build_net`` from the
training split only (no test-set leakage):
  - The paper's 3 edge features (distance, correlation, time-delayed DTW) are
    approximated here as (graph.npz adjacency weight, Pearson correlation,
    DTW distance) between each edge's two node target-series over the train
    split — this codebase no longer carries raw lat/lon (removed along with
    the old pgeocode/knn auto-graph code), so the adjacency weight stands in
    for the distance feature.
  - The VQ commitment loss and an independence/MI term (here: a covariance
    penalty between the invariant representation and the soft environment
    assignment — a light proxy for the paper's mutual-information objective)
    are added as an auxiliary training loss via ``_graph_forward_train``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from st_numeric_baselines.models.gnn.gnn_forecaster import GNNForecasterBase
from st_numeric_baselines.models.registry import register


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Plain O(T^2) dynamic-time-warping distance (T is small: monthly panels)."""
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        row_prev = D[i - 1]
        row = D[i]
        for j in range(1, m + 1):
            cost = abs(ai - b[j - 1])
            row[j] = cost + min(row_prev[j], row[j - 1], row_prev[j - 1])
    return float(D[n, m])


def _zscore(v: torch.Tensor) -> torch.Tensor:
    return (v - v.mean()) / v.std().clamp_min(1e-8)


def _independence_loss(feat: torch.Tensor, env_soft: torch.Tensor) -> torch.Tensor:
    """Covariance penalty between an invariant feature and the soft env assignment."""
    feat_c = feat - feat.mean(dim=0, keepdim=True)
    env_c = env_soft - env_soft.mean(dim=0, keepdim=True)
    n = max(feat.shape[0] - 1, 1)
    cov = (feat_c.t() @ env_c) / n  # [D, n_codes]
    return (cov ** 2).mean()


class VQCodebook(nn.Module):
    """Discretizes a pooled representation into a learned pseudo-environment code."""

    def __init__(self, d_model: int, n_codes: int) -> None:
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(n_codes, d_model) * 0.1)

    def forward(self, z: torch.Tensor):
        dist = torch.cdist(z, self.codebook)  # [B, n_codes]
        idx = dist.argmin(dim=-1)
        z_hard = self.codebook[idx]
        commit_loss = F.mse_loss(z_hard.detach(), z) + F.mse_loss(z_hard, z.detach())
        soft = F.softmax(-dist, dim=-1)
        z_soft = soft @ self.codebook
        return z_hard, z_soft, soft, commit_loss


class CaSTNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        pred_len: int,
        n_nodes: int,
        edge_index: torch.Tensor,   # [2, E]
        edge_feat: torch.Tensor,    # [E, 3]
        *,
        d_model: int = 32,
        n_envs: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pred_len = int(pred_len)
        self.out_dim = int(out_dim)
        self.n_nodes = int(n_nodes)

        self.in_proj = nn.Linear(int(input_dim), int(d_model))
        self.register_buffer("edge_src", edge_index[0].long())
        self.register_buffer("edge_dst", edge_index[1].long())
        self.register_buffer("edge_feat", edge_feat)
        self.edge_gate_mlp = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1))

        self.temporal = nn.GRU(int(d_model), int(d_model), batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.vq = VQCodebook(int(d_model), int(n_envs))
        self.out_proj = nn.Linear(int(d_model) * 2, self.pred_len * self.out_dim)

    def _effective_adj(self, device: torch.device) -> torch.Tensor:
        gate = torch.sigmoid(self.edge_gate_mlp(self.edge_feat)).squeeze(-1)  # [E]
        A = torch.zeros(self.n_nodes, self.n_nodes, device=device, dtype=gate.dtype)
        if self.edge_src.numel() > 0:
            A[self.edge_src, self.edge_dst] = gate
        return A

    def forward(self, x: torch.Tensor):
        B, L, N, _ = x.shape
        h = self.in_proj(x)  # [B, L, N, D]

        A = self._effective_adj(x.device)
        h = torch.einsum("nm,blmd->blnd", A, h) + h  # one-hop spatial propagate + residual

        h_bn = h.permute(0, 2, 1, 3).reshape(B * N, L, -1)
        _, h_last = self.temporal(h_bn)
        z = self.drop(h_last.squeeze(0))  # [B*N, D] — invariant representation

        z_hard, z_soft, soft, commit_loss = self.vq(z)
        if self.training:
            z_q = z + (z_hard - z).detach()  # straight-through estimator
        else:
            z_q = z_soft

        combo = torch.cat([z, z_q], dim=-1)
        out = self.out_proj(combo)  # [B*N, pred_len*out_dim]
        out = out.view(B, N, self.pred_len, self.out_dim).permute(0, 2, 1, 3)

        if self.training:
            aux_loss = commit_loss + _independence_loss(z, soft)
            return out, aux_loss
        return out


@register("cast")
class CaSTForecaster(GNNForecasterBase):
    """CaST forecaster: self-discovered pseudo-environments via a VQ codebook."""

    name: str = "cast"
    d_model: int = 32
    n_envs: int = 10
    dropout: float = 0.1

    def _build_net(self, bundle, n_nodes, *, A_norm, device):
        A_sp = self._A_raw.coalesce()
        idx = A_sp.indices()
        src, dst = idx[0].cpu(), idx[1].cpu()
        weight = A_sp.values().cpu().float()

        train_start, train_end = bundle.raw.split.train
        target_series = bundle.aligned_proc.values[:, train_start:train_end, bundle.raw_target_index]
        ts = torch.from_numpy(np.asarray(target_series)).float()
        ts_c = ts - ts.mean(dim=1, keepdim=True)
        ts_norm = ts_c.pow(2).sum(dim=1).sqrt().clamp_min(1e-8)

        src_l, dst_l = src.tolist(), dst.tolist()
        if src_l:
            corr_vals = torch.tensor([
                float((ts_c[s] * ts_c[d]).sum() / (ts_norm[s] * ts_norm[d] + 1e-8))
                for s, d in zip(src_l, dst_l)
            ])
            ts_np = ts.numpy()
            dtw_vals = torch.tensor([_dtw_distance(ts_np[s], ts_np[d]) for s, d in zip(src_l, dst_l)])
            dtw_sim = -_zscore(dtw_vals)
            edge_feat = torch.stack([_zscore(weight), _zscore(corr_vals), dtw_sim], dim=-1)
        else:
            edge_feat = torch.zeros(0, 3)

        return CaSTNet(
            input_dim=len(bundle.x_cols),
            out_dim=len(bundle.y_cols),
            pred_len=int(bundle.raw.spec.pred_len),
            n_nodes=n_nodes,
            edge_index=torch.stack([src, dst], dim=0),
            edge_feat=edge_feat,
            d_model=int(self.d_model),
            n_envs=int(self.n_envs),
            dropout=float(self.dropout),
        )

    def _graph_forward(self, net, x):
        return net(x)

    def _graph_forward_train(self, net, x, y_true, progress):
        return net(x)
