
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from st_numeric_baselines.graph.torch_adj import spmm_nt


class TemporalConvGLU(nn.Module):

    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        self.c_in = int(c_in)
        self.c_out = int(c_out)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)

        self.conv = nn.Conv2d(
            in_channels=self.c_in,
            out_channels=2 * self.c_out,
            kernel_size=(self.kernel_size, 1),
            bias=True,
        )
        self.res_conv = None
        if self.c_in != self.c_out:
            self.res_conv = nn.Conv2d(self.c_in, self.c_out, kernel_size=(1, 1), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,N,C]
        B, T, N, C = x.shape
        x_c = x.permute(0, 3, 1, 2).contiguous()  # [B,C,T,N]

        # causal left pad on time dimension
        pad = (self.kernel_size - 1)
        if pad > 0:
            x_p = F.pad(x_c, (0, 0, pad, 0)) 
        else:
            x_p = x_c

        out = self.conv(x_p)  # [B,2*C_out,T,N]
        P, Q = out.chunk(2, dim=1)

        res = x_c
        if self.res_conv is not None:
            res = self.res_conv(res)

        y = (P + res) * torch.sigmoid(Q)
        if self.dropout > 0:
            y = F.dropout(y, p=self.dropout, training=self.training)

        y = y.permute(0, 2, 3, 1).contiguous()  # [B,T,N,C_out]
        return y


class ChebGraphConv(nn.Module):

    def __init__(self, c_in: int, c_out: int, K: int = 3):
        super().__init__()
        if K < 1:
            raise ValueError("K must be >= 1")
        self.c_in = int(c_in)
        self.c_out = int(c_out)
        self.K = int(K)
        self.weight = nn.Parameter(torch.empty(self.K, self.c_in, self.c_out))
        self.bias = nn.Parameter(torch.zeros(self.c_out))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # x: [B,T,N,C_in]
        B, T, N, C = x.shape

        # T0 = X
        T0 = x
        outs = []

        if self.K == 1:
            outs = [T0]
        else:
            # Compute AX once
            AX = spmm_nt(A_norm, x)  # [B,T,N,C]
            # With L_tilde=-A: T1 = L_tilde X = -AX
            T1 = -AX
            outs = [T0, T1]

            if self.K >= 3:
                # A^2 X
                A2X = spmm_nt(A_norm, AX)
                # Recurrence gives T2 = 2*A^2 X - X
                T2 = 2.0 * A2X - T0
                outs.append(T2)

            # For K>3, continue recurrence
            Tk_2, Tk_1 = outs[-2], outs[-1]
            for _k in range(3, self.K):
                # Tk = 2 L_tilde Tk-1 - Tk-2 ; L_tilde=-A
                AT = spmm_nt(A_norm, Tk_1)
                Tk = -2.0 * AT - Tk_2
                outs.append(Tk)
                Tk_2, Tk_1 = Tk_1, Tk

        y = 0.0
        for k, Tk in enumerate(outs):
            # Tk @ Wk
            y = y + torch.einsum("btni,io->btno", Tk, self.weight[k])
        y = y + self.bias
        return y


class STConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, *, Kt: int = 3, dropout: float = 0.0):
        super().__init__()
        self.temp1 = TemporalConvGLU(c_in, c_out, kernel_size=Kt, dropout=dropout)
        self.graph = ChebGraphConv(c_out, c_out, K=3)
        self.temp2 = TemporalConvGLU(c_out, c_out, kernel_size=Kt, dropout=dropout)
        self.norm = nn.LayerNorm(c_out)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        x = self.temp1(x)
        x = self.graph(x, A_norm)
        x = self.temp2(x)
        # layer norm on channel dimension
        x = self.norm(x)
        return x


class STGCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        pred_len: int,
        *,
        n_blocks: int = 2,
        Kt: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.pred_len = int(pred_len)
        self.n_blocks = int(n_blocks)

        self.in_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.blocks = nn.ModuleList(
            [STConvBlock(self.hidden_dim, self.hidden_dim, Kt=Kt, dropout=dropout) for _ in range(self.n_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.pred_len),
        )

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # x: [B,L,N,F]
        B, L, N, F = x.shape
        h = self.in_proj(x)  # [B,L,N,H]
        for blk in self.blocks:
            h = blk(h, A_norm)

        # last step
        h_last = h[:, -1, :, :]  # [B,N,H]
        out = self.head(h_last)  # [B,N,pred_len]
        out = out.permute(0, 2, 1).unsqueeze(-1).contiguous()  # [B,pred_len,N,1]
        return out
