from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .frontend_cnn import FrontendCNN
from .encoder_mamba import MambaEncoder
from .predictor import RNNTPredictor
from .joiner import RNNTJoiner


@dataclass
class MCTConfig:
    feat_dim: int = 80
    d_model: int = 256
    n_blocks: int = 6
    state_dim: int = 16
    vocab_size: int = 1024
    joint_dim: int = 320


class MCTModel(nn.Module):
    def __init__(self, cfg: MCTConfig):
        super().__init__()
        self.cfg = cfg
        self.frontend = FrontendCNN(cfg.feat_dim, cfg.d_model)
        self.encoder = MambaEncoder(cfg.d_model, cfg.n_blocks, cfg.state_dim)
        self.predictor = RNNTPredictor(cfg.vocab_size, cfg.d_model, cfg.d_model)
        self.joiner = RNNTJoiner(cfg.d_model, cfg.joint_dim, cfg.vocab_size)

    def forward(self, feats: torch.Tensor, feat_lens: torch.Tensor, tokens: torch.Tensor):
        # feats: (B, T, F), tokens: (B, U)
        enc = self.frontend(feats)
        enc = self.encoder(enc)         # (B, T', D)
        pred = self.predictor(tokens)   # (B, U, D)
        logits = self.joiner(enc, pred) # (B, T', U, V)
        # Derived lengths: T' = floor(T/4) after frontend
        out_lens = torch.clamp(feat_lens // 4, min=1)
        return logits, out_lens
