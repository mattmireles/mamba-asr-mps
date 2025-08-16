"""
Minimal Mamba block using the naive selective_scan for functional baseline.

This module avoids CUDA-specific code and runs with PyTorch MPS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .selective_scan_interface import selective_scan, init_hidden


@dataclass
class MambaConfig:
    d_model: int = 256
    state_dim: int = 16


class MambaBlock(nn.Module):
    """
    Very small Mamba-style block: linear in-proj, selective_scan, linear out-proj.
    """

    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        n = cfg.state_dim

        # Linear projections and parameters
        self.in_proj = nn.Linear(d, d, bias=True)
        self.z_proj = nn.Linear(d, d, bias=True)

        # State-space parameters
        self.A = nn.Parameter(torch.randn(d, n) * 0.01)
        self.D = nn.Parameter(torch.ones(d))
        self.delta_bias = nn.Parameter(torch.zeros(d))

        # Time-varying projections per step will be produced by small conv1d over time
        # Using depthwise conv to keep simple and MPS-friendly
        self.B_conv = nn.Conv1d(d, n, kernel_size=1, bias=True)  # (B,D,L) -> (B,N,L)
        self.C_conv = nn.Conv1d(d, n, kernel_size=1, bias=True)

        self.out_proj = nn.Linear(d, d, bias=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B, L, D = x.shape
        h0 = init_hidden(B, D, self.cfg.state_dim, x.device)

        u = self.in_proj(x)
        z = self.z_proj(x)

        # Produce B and C per step using conv1d over time
        # Rearrange to (B,D,L)
        u_t = u.transpose(1, 2)
        B_proj = self.B_conv(u_t).transpose(1, 2)  # (B,L,N)
        C_proj = self.C_conv(u_t).transpose(1, 2)  # (B,L,N)

        delta = u  # simple parameterization for baseline
        y = selective_scan(
            x=u,
            delta=delta,
            A=self.A,
            B_proj=B_proj,
            C_proj=C_proj,
            D=self.D,
            z=z,
            delta_bias=self.delta_bias,
            h0=h0,
        )
        y = self.out_proj(y)
        return x + self.norm(y)
