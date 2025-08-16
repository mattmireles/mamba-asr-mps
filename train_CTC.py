"""
Minimal CTC training loop for ConMamba on MPS.

Enables MPS CPU fallback to support CTC loss on MPS.
"""
from __future__ import annotations

import os
# Enable CPU fallback for missing MPS ops (e.g., aten::_ctc_loss)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
 # torchaudio is only needed for real data pipelines; keep optional

from modules.Conmamba import ConMambaCTC, ConMambaCTCConfig


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num: int = 32, max_T: int = 800, vocab: int = 1024):
        super().__init__()
        self.num = num
        self.max_T = max_T
        self.vocab = vocab

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        T = torch.randint(low=400, high=self.max_T, size=(1,)).item()
        feats = torch.randn(T, 80)
        feat_len = torch.tensor(T)
        # Random label targets
        tgt_len = torch.randint(low=5, high=50, size=(1,)).item()
        targets = torch.randint(low=1, high=self.vocab - 1, size=(tgt_len,))
        return feats, feat_len, targets


def collate(batch):
    feats_list, feat_lens, targets_list = zip(*batch)
    B = len(batch)
    max_T = max([f.shape[0] for f in feats_list])
    feats = torch.zeros(B, max_T, 80)
    for i, f in enumerate(feats_list):
        feats[i, : f.shape[0]] = f
    feat_lens = torch.stack(feat_lens)
    # Pack targets
    targets = torch.cat(targets_list)
    tgt_lens = torch.tensor([len(t) for t in targets_list], dtype=torch.long)
    return feats, feat_lens, targets, tgt_lens


def train_one_step(model: nn.Module, batch, device: torch.device, criterion, optimizer):
    feats, feat_lens, targets, tgt_lens = batch
    feats = feats.to(device)
    feat_lens = feat_lens.to(device)
    targets = targets.to(device)
    tgt_lens = tgt_lens.to(device)

    logits, out_lens = model(feats, feat_lens)
    # logits: (B, T', V) -> (T', B, V) for CTC
    log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
    loss = criterion(log_probs, targets, out_lens, tgt_lens)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--sanity", action="store_true", help="Run a tiny dummy sanity pass")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    cfg = ConMambaCTCConfig(d_model=256, n_blocks=2, vocab_size=1024)
    model = ConMambaCTC(cfg).to(device)

    if args.sanity:
        ds = DummyDataset(num=8)
    else:
        ds = DummyDataset(num=128)

    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    model.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(dl):
            loss = train_one_step(model, batch, device, criterion, optimizer)
            if step % 10 == 0:
                print(f"epoch {epoch} step {step} loss {loss:.4f}")
        if device.type == "mps":
            torch.mps.synchronize()


if __name__ == "__main__":
    main()
