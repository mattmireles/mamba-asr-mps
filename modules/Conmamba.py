"""
ConMamba CTC model (minimal) for functional MPS baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba.mamba_blocks import MambaBlock, MambaConfig


@dataclass
class ConMambaCTCConfig:
    d_model: int = 256
    n_blocks: int = 4
    vocab_size: int = 1024


class ConMambaCTC(nn.Module):
    def __init__(self, cfg: ConMambaCTCConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # Simple frontend: 1D conv to subsample time
        self.frontend = nn.Sequential(
            nn.Conv1d(in_channels=80, out_channels=d, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(in_channels=d, out_channels=d, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.enc_blocks = nn.ModuleList([MambaBlock(MambaConfig(d_model=d, state_dim=16)) for _ in range(cfg.n_blocks)])
        self.ctc_head = nn.Linear(d, cfg.vocab_size)

    def forward(self, feats: torch.Tensor, feat_lens: torch.Tensor):
        # feats: (B, T, 80) mel-spectrograms
        x = feats.transpose(1, 2)  # (B, 80, T)
        x = self.frontend(x)  # (B, D, T')
        x = x.transpose(1, 2)  # (B, T', D)
        for blk in self.enc_blocks:
            x = blk(x)
        logits = self.ctc_head(x)  # (B, T', V)
        out_lens = torch.clamp(feat_lens // 4, min=1)
        return logits, out_lens
