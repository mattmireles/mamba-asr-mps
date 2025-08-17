"""
Minimal RNNT training loop for MCT on MPS.
- Uses CPU fallback for rnnt_loss if unavailable on MPS/torch
- Falls back to CTC if torchaudio rnnt is not present
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.optim as optim
import time
import contextlib

from modules.mct.mct_model import MCTModel, MCTConfig


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DummyRNNTDataset(torch.utils.data.Dataset):
    def __init__(self, num: int = 16, max_T: int = 600, max_U: int = 40, vocab: int = 1024):
        super().__init__()
        self.num = num
        self.max_T = max_T
        self.max_U = max_U
        self.vocab = vocab

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        T = torch.randint(low=300, high=self.max_T, size=(1,)).item()
        U = torch.randint(low=5, high=self.max_U, size=(1,)).item()
        feats = torch.randn(T, 80)
        feat_len = torch.tensor(T)
        tokens = torch.randint(low=1, high=self.vocab - 1, size=(U,))  # exclude blank 0
        return feats, feat_len, tokens


def collate(batch):
    feats_list, feat_lens, tokens_list = zip(*batch)
    B = len(batch)
    max_T = max([f.shape[0] for f in feats_list])
    max_U = max([t.shape[0] for t in tokens_list])
    feats = torch.zeros(B, max_T, 80)
    tokens = torch.zeros(B, max_U + 1, dtype=torch.long)  # +1 for RNNT start-of-seq blank
    for i, f in enumerate(feats_list):
        feats[i, : f.shape[0]] = f
        # prepend blank 0
        toks = tokens_list[i]
        tokens[i, 1 : 1 + toks.shape[0]] = toks
    feat_lens = torch.stack(feat_lens)
    token_lens = torch.tensor([t.shape[0] + 1 for t in tokens_list], dtype=torch.long)
    return feats, feat_lens, tokens, token_lens


def rnnt_loss_naive_batch(logits: torch.Tensor, tokens: torch.Tensor, out_lens: torch.Tensor, token_lens: torch.Tensor, blank: int = 0) -> torch.Tensor:
    """Very slow, naive RNNT loss for small T,U (debug/sanity only).

    logits: (B, T, U, V) unnormalized
    tokens: (B, U) with tokens[*,0] = blank, labels in 1..V-1
    out_lens: (B,) lengths in T dimension after encoder
    token_lens: (B,) lengths in U dimension including initial blank
    returns mean loss over batch
    """
    B, T_max, U_max, V = logits.shape
    losses = []
    for b in range(B):
        T = int(out_lens[b].item())
        U = int(token_lens[b].item())
        y = tokens[b, 1:U]  # length U-1
        logp = logits[b, :T, :U, :].log_softmax(dim=-1)  # (T,U,V)
        # alpha of shape (T,U)
        alpha = torch.full((T, U), float('-inf'), device=logp.device)
        alpha[0, 0] = 0.0
        for t in range(T):
            for u in range(U):
                a = alpha[t, u]
                if a == float('-inf'):
                    continue
                # blank transition to (t+1,u)
                if t + 1 < T:
                    alpha[t + 1, u] = torch.logaddexp(alpha[t + 1, u], a + logp[t, u, blank])
                # label transition to (t,u+1)
                if u + 1 < U:
                    lbl = int(y[u].item())
                    alpha[t, u + 1] = torch.logaddexp(alpha[t, u + 1], a + logp[t, u, lbl])
        # Termination: add final blank at (T-1,U-1)
        loss_b = -(alpha[T - 1, U - 1] + logp[T - 1, U - 1, blank])
        losses.append(loss_b)
    return torch.stack(losses).mean()

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    cfg = MCTConfig()
    model = MCTModel(cfg).to(device)

    ds = DummyRNNTDataset(num=8 if args.sanity else 64)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    # Try to import rnnt loss
    rnnt_loss = None
    try:
        from torchaudio.prototype.rnnt import rnnt_loss as ta_rnnt_loss  # type: ignore
        rnnt_loss = ta_rnnt_loss
        print("Using torchaudio.prototype.rnnt.rnnt_loss")
    except Exception:
        print("torchaudio rnnt not available; falling back to naive RNNT (slow) for sanity, else CTC(enc) fallback")

    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    model.train()

    # Optional MPS profiling
    try:
        from torch.mps.profiler import profile as mps_profile  # type: ignore
    except Exception:
        mps_profile = contextlib.nullcontext  # type: ignore
    ctx = mps_profile() if args.sanity else contextlib.nullcontext()

    total_frames = 0
    start = time.time()
    with ctx:
        for epoch in range(args.epochs):
            for step, batch in enumerate(dl):
                feats, feat_lens, tokens, token_lens = batch
                total_frames += int(feat_lens.sum().item())
                feats = feats.to(device)
                feat_lens = feat_lens.to(device)
                tokens = tokens.to(device)
                token_lens = token_lens.to(device)

                logits, out_lens = model(feats, feat_lens, tokens)
                # logits: (B, T, U, V)
                if rnnt_loss is not None:
                    log_probs = logits.log_softmax(dim=-1)
                    loss = rnnt_loss(
                        log_probs, tokens, out_lens, token_lens, blank=0, clamp=-1, reduction="mean"
                    )
                else:
                    # Attempt naive RNNT for very small T,U (sanity only)
                    if logits.shape[1] <= 64 and logits.shape[2] <= 16:
                        loss = rnnt_loss_naive_batch(logits, tokens, out_lens, token_lens, blank=0)
                    else:
                        # Fallback: CTC on encoder stream only (approx)
                        enc_only = logits.max(dim=2).values  # (B, T, V)
                        logp = enc_only.log_softmax(dim=-1).transpose(0, 1)
                        targets = []
                        for b in range(tokens.shape[0]):
                            toks = tokens[b, 1 : token_lens[b]]
                            targets.append(toks)
                        flat = torch.cat(targets)
                        tgt_lens = torch.tensor([len(t) for t in targets], device=logp.device)
                        loss = ctc_loss(logp, flat, out_lens, tgt_lens)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                if step % 10 == 0:
                    print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

            if device.type == "mps":
                torch.mps.synchronize()

    elapsed = time.time() - start
    if elapsed > 0:
        print(f"encoder throughput ~ {total_frames/elapsed:.1f} frames/sec (dummy)")


if __name__ == "__main__":
    main()
