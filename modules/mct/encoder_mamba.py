from __future__ import annotations

import torch
import torch.nn as nn

from ..mamba.mamba_blocks import MambaBlock, MambaConfig


class MambaEncoder(nn.Module):
    def __init__(self, d_model: int = 256, n_blocks: int = 6, state_dim: int = 16):
        super().__init__()
        self.blocks = nn.ModuleList([MambaBlock(MambaConfig(d_model=d_model, state_dim=state_dim)) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x
