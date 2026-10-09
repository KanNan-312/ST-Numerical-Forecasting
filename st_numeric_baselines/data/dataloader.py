
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from .dataset import SampleMeta


def _pad_left(x: torch.Tensor, pad_len: int, value: float = 0.0) -> torch.Tensor:
    if pad_len <= 0:
        return x
    pad = torch.full((pad_len, x.shape[1]), float(value), dtype=x.dtype)
    return torch.cat([pad, x], dim=0)


def _pad_right(x: torch.Tensor, pad_len: int, value: float = 0.0) -> torch.Tensor:
    if pad_len <= 0:
        return x
    pad = torch.full((pad_len, x.shape[1]), float(value), dtype=x.dtype)
    return torch.cat([x, pad], dim=0)


def make_collate_fn(
    *,
    pad_to: Optional[int] = None,
    pad_side: str = "left",
    return_meta_dict: bool = True,
) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    pad_side = pad_side.lower()
    if pad_side not in ("left", "right"):
        raise ValueError("pad_side must be 'left' or 'right'")

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        xs = [b["x"] for b in batch]
        ys = [b["y"] for b in batch]
        xms = [b["x_mark"] for b in batch]
        yms = [b["y_mark"] for b in batch]
        metas = [b["meta"] for b in batch]

        # pad x/x_mark if needed
        if pad_to is not None:
            cur_len = xs[0].shape[0]
            if cur_len > pad_to:
                raise ValueError(f"pad_to={pad_to} smaller than seq_len={cur_len}")
            pad_len = pad_to - cur_len

            if pad_len > 0:
                pad_fn = _pad_left if pad_side == "left" else _pad_right
                xs = [pad_fn(x, pad_len, value=0.0) for x in xs]
                xms = [pad_fn(xm, pad_len, value=0.0) for xm in xms]

                # mask: 1 for real tokens, 0 for padding
                # shape [B, pad_to]
                mask = torch.ones((len(batch), pad_to), dtype=torch.float32)
                if pad_side == "left":
                    mask[:, :pad_len] = 0.0
                else:
                    mask[:, -pad_len:] = 0.0
            else:
                mask = torch.ones((len(batch), cur_len), dtype=torch.float32)
        else:
            mask = torch.ones((len(batch), xs[0].shape[0]), dtype=torch.float32)

        x = torch.stack(xs, dim=0)          # [B, L, Dx]
        y = torch.stack(ys, dim=0)          # [B, Ly, Dy]
        x_mark = torch.stack(xms, dim=0)    # [B, L, 2]
        y_mark = torch.stack(yms, dim=0)    # [B, Ly, 2]

        if return_meta_dict:
            meta_out = [asdict(m) if isinstance(m, SampleMeta) else dict(m) for m in metas]
        else:
            meta_out = metas

        return {"x": x, "y": y, "x_mark": x_mark, "y_mark": y_mark, "x_mask": mask, "meta": meta_out}

    return collate


def make_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pad_to: Optional[int] = None,
    pad_side: str = "left",
    drop_last: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    collate_fn = make_collate_fn(pad_to=pad_to, pad_side=pad_side, return_meta_dict=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
