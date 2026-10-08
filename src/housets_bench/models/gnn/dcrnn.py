"""DCRNN — Diffusion Convolutional Recurrent Neural Network (Li et al., ICLR 2018).

Reference: https://github.com/liyaguang/DCRNN

Seq2seq encoder-decoder built from diffusion-convolutional GRU cells. Diffusion
convolution replaces the GRU's linear gates with a bidirectional random-walk
graph convolution: a forward pass (row-normalized ``A``) and a backward pass
(row-normalized ``A.T``), both derived here from the single adjacency supplied
via ``graph.npz`` (no separate graph asset needed — matches the DCRNN paper,
which also derives both directions from one adjacency matrix).

The decoder uses inverse-sigmoid scheduled sampling during training (feed the
ground-truth previous step vs. the model's own prediction, with a decaying
probability) via :meth:`DCRNNForecaster._graph_forward_train`; eval/predict are
always free-running (:meth:`DCRNNForecaster._graph_forward`), so no ground truth
ever leaks into evaluation.

Traffic data (confirmed from liyaguang/DCRNN's own ``generate_training_data.py``):
the original adds time-of-day as a plain extra **input channel** (no
model-side embedding at all), computed in the data-prep script, not the
network. No model change is needed here for that — the traffic data loaders
(``housets_bench.data.io.load_metr_la``/``load_pems08``) already emit a
``time_of_day`` column; just include it via the dataset config's
``feature_cols`` (e.g. ``feature_cols: [time_of_day]``) and it flows into
this model as an ordinary extra channel of ``bundle.x_cols``, exactly
matching the original convention.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn

from housets_bench.graph.torch_adj import normalize_adj_random_walk
from housets_bench.models.gnn.gnn_forecaster import GNNForecasterBase
from housets_bench.models.registry import register


class DiffusionConv(nn.Module):
    """K-hop diffusion convolution over one or more supports ``[S, N, N]`` (dense)."""

    def __init__(self, c_in: int, c_out: int, supports: torch.Tensor, max_step: int = 2) -> None:
        super().__init__()
        if max_step < 1:
            raise ValueError("max_step must be >= 1")
        self.max_step = int(max_step)
        self.register_buffer("supports", supports)  # [S, N, N]
        n_supports = supports.shape[0]
        c_cat = c_in * (1 + n_supports * self.max_step)
        self.linear = nn.Linear(c_cat, c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, c_in]
        outs = [x]
        for s in range(self.supports.shape[0]):
            A = self.supports[s]
            x1 = torch.einsum("nm,bmc->bnc", A, x)
            outs.append(x1)
            for _ in range(2, self.max_step + 1):
                x1 = torch.einsum("nm,bmc->bnc", A, x1)
                outs.append(x1)
        h = torch.cat(outs, dim=-1)
        return self.linear(h)


class DCGRUCell(nn.Module):
    """Diffusion Convolutional GRU cell: reset/update/candidate gates via diffusion conv."""

    def __init__(self, input_dim: int, hidden_dim: int, supports: torch.Tensor, max_step: int = 2) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        c_in = int(input_dim) + int(hidden_dim)
        self.gate_conv = DiffusionConv(c_in, 2 * int(hidden_dim), supports, max_step)
        self.cand_conv = DiffusionConv(c_in, int(hidden_dim), supports, max_step)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x: [B, N, Dx]   h: [B, N, H]
        xh = torch.cat([x, h], dim=-1)
        gates = torch.sigmoid(self.gate_conv(xh))
        r, u = gates.chunk(2, dim=-1)
        xh_r = torch.cat([x, r * h], dim=-1)
        c = torch.tanh(self.cand_conv(xh_r))
        return u * h + (1.0 - u) * c


class DCRNNNet(nn.Module):
    """Seq2seq DCGRU encoder-decoder.

    forward(x, y_true=None, teacher_forcing_ratio=0.0) -> [B, pred_len, N, out_dim]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        pred_len: int,
        n_layers: int,
        supports: torch.Tensor,
        max_step: int = 2,
    ) -> None:
        super().__init__()
        self.pred_len = int(pred_len)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)

        self.encoder_cells = nn.ModuleList([
            DCGRUCell(input_dim if i == 0 else hidden_dim, hidden_dim, supports, max_step)
            for i in range(self.n_layers)
        ])
        self.decoder_cells = nn.ModuleList([
            DCGRUCell(out_dim if i == 0 else hidden_dim, hidden_dim, supports, max_step)
            for i in range(self.n_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def _encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        B, L, N, _ = x.shape
        h = [x.new_zeros(B, N, self.hidden_dim) for _ in range(self.n_layers)]
        for t in range(L):
            inp = x[:, t]
            for i, cell in enumerate(self.encoder_cells):
                h[i] = cell(inp, h[i])
                inp = h[i]
        return h

    def _decode(
        self,
        h: List[torch.Tensor],
        *,
        batch_size: int,
        n_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
        y_true: Optional[torch.Tensor],
        teacher_forcing_ratio: float,
    ) -> torch.Tensor:
        dec_input = torch.zeros(batch_size, n_nodes, self.out_dim, device=device, dtype=dtype)
        outs = []
        for t in range(self.pred_len):
            inp = dec_input
            for i, cell in enumerate(self.decoder_cells):
                h[i] = cell(inp, h[i])
                inp = h[i]
            y_t = self.out_proj(h[-1])  # [B, N, out_dim]
            outs.append(y_t)
            use_truth = (
                y_true is not None
                and teacher_forcing_ratio > 0.0
                and bool(torch.rand(()) < teacher_forcing_ratio)
            )
            dec_input = y_true[:, t] if use_truth else y_t
        return torch.stack(outs, dim=1)  # [B, pred_len, N, out_dim]

    def forward(
        self,
        x: torch.Tensor,
        y_true: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        B, L, N, _ = x.shape
        h = self._encode(x)
        return self._decode(
            h,
            batch_size=B,
            n_nodes=N,
            device=x.device,
            dtype=x.dtype,
            y_true=y_true,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )


@register("dcrnn")
class DCRNNForecaster(GNNForecasterBase):
    """DCRNN forecaster with inverse-sigmoid scheduled sampling."""

    name: str = "dcrnn"
    hidden_dim: int = 32
    n_layers: int = 2
    max_diffusion_step: int = 2
    cl_decay_rate: float = 10.0  # inverse-sigmoid decay rate for teacher forcing, in progress-space [0,1]

    def _build_net(self, bundle, n_nodes, *, A_norm, device):
        A_sp = self._A_raw.to(device)
        P_f = normalize_adj_random_walk(A_sp).to_dense()
        P_b = normalize_adj_random_walk(A_sp.transpose(0, 1).coalesce()).to_dense()
        supports = torch.stack([P_f, P_b], dim=0)

        return DCRNNNet(
            input_dim=len(bundle.x_cols),
            hidden_dim=int(self.hidden_dim),
            out_dim=len(bundle.y_cols),
            pred_len=int(bundle.raw.spec.pred_len),
            n_layers=int(self.n_layers),
            supports=supports,
            max_step=int(self.max_diffusion_step),
        )

    def _sampling_prob(self, progress: float) -> float:
        # inverse-sigmoid decay over training progress in [0,1]: ~1 at the start,
        # decaying towards 0 by the end (same shape as the paper's k/(k+exp(step/k)),
        # reparametrized in progress-space since only the training-progress fraction
        # is threaded through the shared GNNForecasterBase training loop).
        k = float(self.cl_decay_rate)
        return k / (k + math.exp(progress * k))

    def _graph_forward(self, net, x):
        return net(x, y_true=None, teacher_forcing_ratio=0.0)

    def _graph_forward_train(self, net, x, y_true, progress):
        ratio = self._sampling_prob(progress)
        return net(x, y_true=y_true, teacher_forcing_ratio=ratio)
