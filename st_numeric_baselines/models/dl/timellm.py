"""Time-LLM: Time Series Forecasting by Reprogramming Large Language Models.

Jin et al., ICLR 2024 — https://github.com/KimMeen/Time-LLM

Key idea: a lightweight *Reprogramming Layer* (cross-attention) translates
patch embeddings into the LLM's token vocabulary space by attending over a
small set of learnable prototype vectors. A brief task-description prompt is
prepended. The frozen LLM then processes [prompt | reprogrammed patches]; a
linear head maps the full sequence hidden states to the forecast horizon.

Default backbone: GPT-2 (practical, no special download access required).
The original paper uses LLaMA-7B; set ``llm_model`` and ``llm_path`` in
hparams to switch to a local LLaMA checkpoint.

Requires: pip install transformers
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from st_numeric_baselines.bundles.datatypes import ProcBundle
from st_numeric_baselines.models.base import BaseForecaster
from st_numeric_baselines.models.registry import register

try:
    from transformers import GPT2Config, GPT2Model
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


class _ReprogrammingLayer(nn.Module):
    """Cross-attention that maps patch embeddings → LLM vocabulary space.

    Queries come from patch embeddings; Keys/Values come from a small set of
    learnable prototype vectors (standing in for representative word embeddings).
    This produces tokens that "speak the LLM's language" without requiring the
    patch dimension to match the LLM embedding dim exactly.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_keys: int,
        d_llm: int,
        n_prototypes: int,
    ) -> None:
        super().__init__()
        self.n_heads = int(n_heads)
        self.d_keys = int(d_keys)

        self.q_proj = nn.Linear(int(d_model), int(n_heads) * int(d_keys))
        self.k_proj = nn.Linear(int(d_llm), int(n_heads) * int(d_keys))
        self.v_proj = nn.Linear(int(d_llm), int(n_heads) * int(d_keys))
        self.out_proj = nn.Linear(int(n_heads) * int(d_keys), int(d_llm))

        # Learnable prototype vocabulary
        self.prototypes = nn.Parameter(torch.randn(int(n_prototypes), int(d_llm)) * 0.02)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        # patches: [B, N, d_model]
        B, N, _ = patches.shape
        H, Dk = self.n_heads, self.d_keys

        q = self.q_proj(patches).view(B, N, H, Dk).permute(0, 2, 1, 3)   # [B, H, N, Dk]

        proto = self.prototypes                                             # [P, d_llm]
        k = self.k_proj(proto).view(-1, H, Dk).permute(1, 0, 2).unsqueeze(0)  # [1, H, P, Dk]
        v = self.v_proj(proto).view(-1, H, Dk).permute(1, 0, 2).unsqueeze(0)

        scale = Dk ** -0.5
        attn = F.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)         # [B, H, N, P]
        y = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, H * Dk)          # [B, N, H*Dk]
        return self.out_proj(y)                                             # [B, N, d_llm]


class _TimeLLMNet(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        pred_len: int,
        input_dim: int,
        out_dim: int,
        patch_len: int,
        stride: int,
        d_model: int,
        n_heads: int,
        d_keys: int,
        n_prototypes: int,
        prompt_len: int,
        gpt_layers: int,
        dropout: float,
        pretrained: bool,
    ) -> None:
        super().__init__()
        if not _HAS_TRANSFORMERS:
            raise ImportError(
                "TimeLLMForecaster requires the `transformers` package.\n"
                "Install it with:  pip install transformers"
            )

        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.out_dim = int(out_dim)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.prompt_len = int(prompt_len)

        patch_num = (int(seq_len) - int(patch_len)) // int(stride) + 1
        self.patch_num = patch_num
        D_LLM = 768  # GPT-2 hidden size

        # 1. Patch embedding (patch_len × Dx → d_model)
        self.patch_embed = nn.Linear(int(patch_len) * int(input_dim), int(d_model))

        # 2. Reprogramming layer: d_model → D_LLM
        self.reprogramming = _ReprogrammingLayer(
            d_model=int(d_model),
            n_heads=int(n_heads),
            d_keys=int(d_keys),
            d_llm=D_LLM,
            n_prototypes=int(n_prototypes),
        )

        # 3. Learnable prompt prefix (task description in LLM token space)
        self.prompt = nn.Parameter(torch.randn(1, int(prompt_len), D_LLM) * 0.02)

        # 4. Frozen LLM backbone
        if pretrained:
            self.llm = GPT2Model.from_pretrained("gpt2")
        else:
            cfg = GPT2Config(n_embd=D_LLM, n_layer=max(1, int(gpt_layers)), n_head=12)
            self.llm = GPT2Model(cfg)

        self.llm.h = self.llm.h[: int(gpt_layers)]

        for param in self.llm.parameters():
            param.requires_grad = False
        for name, param in self.llm.named_parameters():
            if "ln_" in name:
                param.requires_grad = True

        self.dropout = nn.Dropout(float(dropout))

        # 5. Output projection: flatten all token hidden states → pred_len × out_dim
        total_tokens = int(prompt_len) + patch_num
        self.out_layer = nn.Linear(D_LLM * total_tokens, int(pred_len) * int(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, Dx]
        B, L, Dx = x.shape
        if L != self.seq_len:
            if L < self.seq_len:
                raise ValueError(f"Input length L={L} < seq_len={self.seq_len}")
            x = x[:, -self.seq_len :, :]

        # Per-variate instance normalisation
        means = x.mean(dim=1, keepdim=True).detach()
        x = x - means
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / std

        # Patch extraction
        xp = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        _, N, Dx2, P = xp.shape
        xp = xp.reshape(B, N, Dx2 * P)       # [B, N, Dx*patch_len]

        # Patch embedding → reprogramming → LLM space
        xp = self.patch_embed(xp)             # [B, N, d_model]
        xp = self.reprogramming(xp)           # [B, N, D_LLM]

        # Prepend task prompt
        prompt = self.prompt.expand(B, -1, -1)
        tokens = torch.cat([prompt, xp], dim=1)   # [B, prompt_len + N, D_LLM]
        tokens = self.dropout(tokens)

        # LLM forward
        out = self.llm(inputs_embeds=tokens).last_hidden_state   # [B, T, D_LLM]

        # Project to forecast
        out = out.reshape(B, -1)              # [B, T * D_LLM]
        out = self.out_layer(out)             # [B, pred_len * out_dim]
        out = out.view(B, self.pred_len, self.out_dim)

        # Denormalise
        out = out * std[:, :1, : self.out_dim] + means[:, :1, : self.out_dim]
        return out


@register("timellm")
class TimeLLMForecaster(BaseForecaster):
    """Time-LLM: reprogrammes a frozen GPT-2 backbone for forecasting (ICLR 2024).

    Uses a cross-attention Reprogramming Layer to translate patch embeddings
    into the LLM's vocabulary space, then prepends a learnable task-prompt.
    Default backbone is GPT-2; the original paper uses LLaMA-7B.

    Requires the ``transformers`` library (``pip install transformers``).
    """

    name: str = "timellm"

    # model hyper-parameters
    patch_len: int = 3
    stride: int = 1
    d_model: int = 64        # patch embedding dim (before reprogramming)
    n_heads: int = 4         # reprogramming attention heads
    d_keys: int = 32         # per-head key/value dim in reprogramming
    n_prototypes: int = 64   # number of learnable prototype word embeddings
    prompt_len: int = 8      # number of learnable prompt tokens prepended
    gpt_layers: int = 6      # number of GPT-2 transformer blocks to use

    dropout: float = 0.1
    pretrained: bool = True

    # training hyper-parameters
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    patience: int = 3
    max_train_batches: Optional[int] = None
    seed: int = 0

    def __init__(self) -> None:
        self._net: Optional[_TimeLLMNet] = None
        self.train_history: list[dict[str, Any]] = []

    def fit(self, bundle: ProcBundle, *, device: Optional[torch.device] = None) -> None:
        if not _HAS_TRANSFORMERS:
            raise ImportError(
                "TimeLLMForecaster requires the `transformers` package.\n"
                "Install it with:  pip install transformers"
            )

        dev = device or torch.device("cpu")
        torch.manual_seed(int(self.seed))

        train_dl = bundle.dataloaders["train"]
        val_dl = bundle.dataloaders["val"]
        Dx = int(len(bundle.x_cols))
        Dy = int(len(bundle.y_cols))
        seq_len = int(bundle.raw.spec.seq_len)
        pred_len = int(bundle.raw.spec.pred_len)

        patch_len = int(self.patch_len)
        stride = int(self.stride)
        if patch_len > seq_len:
            patch_len = seq_len
            stride = seq_len

        net = _TimeLLMNet(
            seq_len=seq_len,
            pred_len=pred_len,
            input_dim=Dx,
            out_dim=Dy,
            patch_len=patch_len,
            stride=stride,
            d_model=int(self.d_model),
            n_heads=int(self.n_heads),
            d_keys=int(self.d_keys),
            n_prototypes=int(self.n_prototypes),
            prompt_len=int(self.prompt_len),
            gpt_layers=int(self.gpt_layers),
            dropout=float(self.dropout),
            pretrained=bool(self.pretrained),
        ).to(dev)

        trainable = [p for p in net.parameters() if p.requires_grad]
        opt = torch.optim.Adam(trainable, lr=float(self.lr), weight_decay=float(self.weight_decay))

        best_val = math.inf
        best_state: Optional[Dict[str, torch.Tensor]] = None
        bad_epochs = 0
        self.train_history = []

        _max_bt = int(self.max_train_batches) if self.max_train_batches is not None else None
        _train_total = min(len(train_dl), _max_bt) if _max_bt is not None else len(train_dl)

        epoch_bar = tqdm(range(int(self.epochs)), desc=f"[{self.name}]", unit="ep")
        for ep in epoch_bar:
            t0 = time.perf_counter()
            net.train()
            tr_sse, tr_n = 0.0, 0
            tr_bar = tqdm(train_dl, desc="  train", total=_train_total, leave=False, unit="bt")
            for bi, batch in enumerate(tr_bar):
                if _max_bt is not None and bi >= _max_bt:
                    tr_bar.close()
                    break
                x = batch["x"].to(dev)
                y_true = batch["y"][:, -pred_len:, :].to(dev)
                y_pred = net(x)
                loss = F.mse_loss(y_pred, y_true)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if float(self.grad_clip) > 0:
                    nn.utils.clip_grad_norm_(net.parameters(), max_norm=float(self.grad_clip))
                opt.step()
                tr_sse += float(loss.detach()) * int(y_true.numel())
                tr_n += int(y_true.numel())
                tr_bar.set_postfix({"loss": f"{float(loss.detach()):.4g}"})

            net.eval()
            v_sse, v_n = 0.0, 0
            with torch.no_grad():
                v_bar = tqdm(val_dl, desc="  val  ", leave=False, unit="bt")
                for batch in v_bar:
                    x = batch["x"].to(dev)
                    y_true = batch["y"][:, -pred_len:, :].to(dev)
                    y_pred = net(x)
                    diff = (y_pred - y_true).float()
                    v_sse += float((diff * diff).sum())
                    v_n += int(diff.numel())
                    v_bar.set_postfix({"mse": f"{v_sse / max(v_n, 1):.4g}"})

            tr_mse = tr_sse / max(tr_n, 1)
            val_mse = v_sse / max(v_n, 1) if v_n > 0 else float("inf")
            self.train_history.append({
                "epoch": ep + 1, "train_mse": tr_mse, "val_mse": val_mse,
                "epoch_time_sec": time.perf_counter() - t0,
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
            epoch_bar.set_postfix({"train": f"{tr_mse:.4g}", "val": f"{val_mse:.4g}", "best": f"{best_val:.4g}"})

        if best_state is not None:
            net.load_state_dict(best_state)
        self._net = net

    def predict_batch(
        self,
        batch: Dict[str, Any],
        *,
        bundle: ProcBundle,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError("TimeLLMForecaster must be fit() before predict_batch()")
        dev = device or next(self._net.parameters()).device
        self._net.to(dev).eval()
        with torch.no_grad():
            return self._net(batch["x"].to(dev))
