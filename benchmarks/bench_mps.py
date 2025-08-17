from __future__ import annotations

import os
import time
import contextlib

import torch
import torch.nn as nn

from modules.Conmamba import ConMambaCTC, ConMambaCTCConfig
from modules.mct.mct_model import MCTModel, MCTConfig


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def bench_ctc(device: torch.device, steps: int = 20, batch_size: int = 2, T: int = 800):
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    model = ConMambaCTC(ConMambaCTCConfig()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    crit = nn.CTCLoss(blank=0, zero_infinity=True)

    feats = torch.randn(batch_size, T, 80, device=device)
    feat_lens = torch.full((batch_size,), T, dtype=torch.long, device=device)
    targets = torch.randint(1, 1023, (batch_size * 20,), device=device)
    tgt_lens = torch.full((batch_size,), 20, dtype=torch.long, device=device)

    total = 0
    start = time.time()
    for i in range(steps):
        logits, out_lens = model(feats, feat_lens)
        logp = logits.log_softmax(dim=-1).transpose(0, 1)
        loss = crit(logp, targets, out_lens, tgt_lens)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += int(feat_lens.sum().item())
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.time() - start
    print(f"CTC: steps={steps} bs={batch_size} T={T} throughput~{total/elapsed:.1f} frames/s loss={loss.item():.3f}")


def bench_rnnt(device: torch.device, steps: int = 10, batch_size: int = 2, T: int = 600, U: int = 30):
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    model = MCTModel(MCTConfig()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    crit = nn.CTCLoss(blank=0, zero_infinity=True)

    feats = torch.randn(batch_size, T, 80, device=device)
    feat_lens = torch.full((batch_size,), T, dtype=torch.long, device=device)
    tokens = torch.randint(1, 1023, (batch_size, U), device=device)
    token_lens = torch.full((batch_size,), U + 1, dtype=torch.long, device=device)

    total = 0
    start = time.time()
    for i in range(steps):
        logits, out_lens = model(feats, feat_lens, torch.cat([torch.zeros(batch_size,1,dtype=torch.long,device=device), tokens], dim=1))
        # Encoder-only CTC fallback approximation
        enc_only = logits.max(dim=2).values  # (B, T, V)
        logp = enc_only.log_softmax(dim=-1).transpose(0, 1)
        # Flatten targets
        flat = tokens.reshape(-1)
        tgt_lens = torch.full((batch_size,), U, dtype=torch.long, device=device)
        loss = crit(logp, flat, out_lens, tgt_lens)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += int(feat_lens.sum().item())
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.time() - start
    print(f"RNNT(enc-ctc): steps={steps} bs={batch_size} T={T} U={U} throughput~{total/elapsed:.1f} frames/s loss={loss.item():.3f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--bs", type=int, default=2)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    try:
        from torch.mps.profiler import profile as mps_profile  # type: ignore
    except Exception:
        mps_profile = contextlib.nullcontext  # type: ignore
    ctx = mps_profile() if args.profile else contextlib.nullcontext()
    with ctx:
        bench_ctc(device, steps=5, batch_size=args.bs, T=800)
        bench_rnnt(device, steps=3, batch_size=args.bs, T=600, U=30)


if __name__ == "__main__":
    main()
