from __future__ import annotations

import torch
import torch.nn as nn


class RNNTPredictor(nn.Module):
    """Simple RNNT predictor network: Embedding + single-layer GRU."""

    def __init__(self, vocab_size: int, d_model: int = 256, embed_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=d_model, batch_first=True)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, U)
        x = self.embedding(tokens)
        x, _ = self.gru(x)
        x = self.proj(x)
        return self.norm(x)
