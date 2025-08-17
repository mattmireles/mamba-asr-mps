from __future__ import annotations

import torch
import torch.nn as nn


class FrontendCNN(nn.Module):
    """ANE-friendly simple CNN frontend with 4x time subsampling.

    Input: (B, T, F=80) mel features
    Output: (B, T/4, D)
    """

    def __init__(self, feat_dim: int = 80, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(feat_dim, d_model, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, T, F)
        x = feats.transpose(1, 2)  # (B, F, T)
        x = self.net(x)            # (B, D, T')
        x = x.transpose(1, 2)      # (B, T', D)
        return self.norm(x)
