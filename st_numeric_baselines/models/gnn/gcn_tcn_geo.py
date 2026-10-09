from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeoGCN_TCN(nn.Module):
    """Minimal GCN + TCN model for full-graph forecasting.

    Inputs
    ------
    x: [B, L, N, F_in]
    A_norm: torch sparse [N, N] (normalized adjacency)

    Output
    ------
    y_hat: [B, H, N, 1] in processed/log space
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        pred_len: int,
        dropout: float = 0.1,
        tcn_kernel: int = 3,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.pred_len = int(pred_len)
        self.dropout = float(dropout)
        self.tcn_kernel = int(tcn_kernel)

        self.in_proj = nn.Linear(self.input_dim, self.hidden_dim)

        # temporal conv per node: Conv1d over time dimension
        k = self.tcn_kernel
        if k % 2 == 0:
            k += 1
        pad = k // 2
        self.tcn = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=k, padding=pad),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.out_proj = nn.Linear(self.hidden_dim, self.pred_len)

    def _gcn_step(self, x_t: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """One GCN aggregation for a single time step.

        x_t: [B, N, H]
        returns: [B, N, H]
        """
        B, N, H = x_t.shape
        outs = []
        # sparse.mm supports [N,N] @ [N,H] -> [N,H]
        for b in range(B):
            outs.append(torch.sparse.mm(A_norm, x_t[b]))
        out = torch.stack(outs, dim=0)  # [B,N,H]
        return out

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        B, L, N, Fin = x.shape
        h = self.in_proj(x)  # [B,L,N,H]
        h = F.relu(h)

        # graph conv at each time step
        h_list = []
        for t in range(L):
            ht = h[:, t, :, :]  # [B,N,H]
            ht = self._gcn_step(ht, A_norm)
            ht = F.relu(ht)
            ht = F.dropout(ht, p=self.dropout, training=self.training)
            h_list.append(ht)
        h_seq = torch.stack(h_list, dim=1)  # [B,L,N,H]

        # temporal conv per node
        # [B,L,N,H] -> [B,N,H,L] -> [B*N,H,L]
        h_tcn_in = h_seq.permute(0, 2, 3, 1).contiguous().view(B * N, self.hidden_dim, L)
        h_tcn_out = self.tcn(h_tcn_in)  # [B*N,H,L]
        h_last = h_tcn_out[:, :, -1]    # [B*N,H]

        # predict H steps (channels) for each node
        y = self.out_proj(h_last)       # [B*N, pred_len]
        y = y.view(B, N, self.pred_len).permute(0, 2, 1).unsqueeze(-1).contiguous()  # [B,H,N,1]
        return y
