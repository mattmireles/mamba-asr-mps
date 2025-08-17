from __future__ import annotations

import torch
import torch.nn as nn


class RNNTJoiner(nn.Module):
    """Additive joiner: project encoder and predictor to joint dim, tanh, then vocab."""

    def __init__(self, d_model: int = 256, joint_dim: int = 320, vocab_size: int = 1024):
        super().__init__()
        self.enc_proj = nn.Linear(d_model, joint_dim)
        self.pred_proj = nn.Linear(d_model, joint_dim)
        self.activation = nn.Tanh()
        self.out = nn.Linear(joint_dim, vocab_size)

    def forward(self, enc: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        # enc: (B, T, D), pred: (B, U, D)
        e = self.enc_proj(enc).unsqueeze(2)   # (B, T, 1, J)
        p = self.pred_proj(pred).unsqueeze(1) # (B, 1, U, J)
        z = self.activation(e + p)            # (B, T, U, J)
        logits = self.out(z)                  # (B, T, U, V)
        return logits
