#!/usr/bin/env python3
"""
Extract a 1024→29 projection matrix from a PyTorch checkpoint.

Usage:
  python Mamba-ASR-MPS/scripts/extract_projection_from_ckpt.py \
    --ckpt /path/to/checkpoint.pt \
    --w-key model.proj.weight \
    --b-key model.proj.bias \
    --out Mamba-ASR-MPS/exports/projection_1024x29.csv

Notes:
- Expects weight of shape (29, 1024) and optional bias of shape (29,).
- Writes a CSV with 1024 rows × 29 cols, entries in log-space:
    P[i,k] = log_softmax(W[k, i] + b[k]) over k
- If bias missing, treated as zeros.
- If weight transposed (1024, 29), pass --transpose-w.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to PyTorch checkpoint (.pt/.pth)")
    ap.add_argument("--w-key", required=True, help="State dict key for projection weight (29 x 1024 or 1024 x 29 with --transpose-w)")
    ap.add_argument("--b-key", default=None, help="State dict key for projection bias (29,). Optional")
    ap.add_argument("--transpose-w", action="store_true", help="If set, transpose weight before extraction (from 1024x29 to 29x1024)")
    ap.add_argument("--out", required=True, help="Output CSV path (1024 rows x 29 cols, log-space)")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    # Load flexibly
    obj = torch.load(str(ckpt_path), map_location="cpu")
    state = obj.get("state_dict", obj)
    if not isinstance(state, dict):
        raise SystemExit("Could not find state dict in checkpoint (expected dict or {'state_dict': dict})")

    if args.w_key not in state:
        keys = "\n  - ".join(state.keys())
        raise SystemExit(f"Weight key not found: {args.w_key}\nAvailable keys:\n  - {keys}")

    W = state[args.w_key].float()
    if args.transpose_w:
        W = W.t()
    if W.shape != (29, 1024):
        raise SystemExit(f"Weight shape must be (29,1024) after transpose handling, got {tuple(W.shape)}")

    if args.b_key is not None:
        if args.b_key not in state:
            keys = "\n  - ".join(state.keys())
            raise SystemExit(f"Bias key not found: {args.b_key}\nAvailable keys:\n  - {keys}")
        b = state[args.b_key].float()
        if b.shape != (29,):
            raise SystemExit(f"Bias shape must be (29,), got {tuple(b.shape)}")
    else:
        b = torch.zeros(29, dtype=torch.float32)

    # Compute column-wise log-softmax: for each i in 0..1023, P[i,:] = log_softmax(W[:,i] + b)
    logits = W + b[:, None]  # (29, 1024)
    P = F.log_softmax(logits, dim=0).t().contiguous()  # (1024, 29)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for i in range(P.shape[0]):
            row = ",".join(f"{float(x):.8f}" for x in P[i].tolist())
            f.write(row + "\n")
    print(f"Wrote projection CSV: {out_path}")


if __name__ == "__main__":
    main()
