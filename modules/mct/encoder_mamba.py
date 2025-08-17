"""
Mamba encoder module for efficient sequence modeling in speech recognition.

This module implements a stack of Mamba blocks for acoustic sequence modeling
in speech recognition systems. It provides linear complexity alternative to
Transformer encoders while maintaining comparable modeling capacity.

Architectural Role:
- Called by: MCTModel.forward() for acoustic sequence encoding
- Used in: RNN-T and CTC speech recognition pipelines  
- Core component: MambaBlock stack from mamba_blocks.py
- Integration: Part of encoder-predictor-joiner RNN-T architecture

Mamba Advantages for Speech:
- Linear complexity O(L) vs. Transformer O(L²) for long audio
- Selective state space modeling for content-aware processing
- Efficient streaming inference capability
- Reduced memory requirements for long sequences

Apple Silicon Optimization:
- Each MambaBlock uses MPS-compatible operations
- Sequential processing minimizes memory allocation overhead
- Unified memory architecture enables deeper stacks
- Profiling-friendly for performance analysis

Performance Characteristics:
- Dominated by selective_scan operations in each block
- Linear scaling with number of blocks
- Memory usage: O(B * L * D * n_blocks + B * D * N * n_blocks)
- Computational complexity: O(B * L * D * N * n_blocks)

Called By:
- MCTModel.forward() in mct_model.py for RNN-T encoding
- Inference pipelines for streaming speech recognition
- Training scripts via MCTModel instantiation

Integration Points:
- Input: Frontend-processed acoustic features (B, T/4, D)
- Output: Encoded sequence representations (B, T/4, D)
- Used with: RNNTPredictor and RNNTJoiner for complete pipeline

References:
- Mamba: Gu & Dao Selective State Space Models
- Speech modeling: Linear complexity for long audio sequences
- Apple Silicon optimization: README/Mamba-on-Apple-Silicon.md
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..mamba.mamba_blocks import MambaBlock, MambaConfig


class MambaEncoder(nn.Module):
    """Stack of Mamba blocks for acoustic sequence encoding in speech recognition.
    
    This encoder provides the core sequence modeling component for speech
    recognition systems, replacing Transformer encoders with Mamba's linear
    complexity selective state space models.
    
    Architecture Design:
    - Sequential stack of MambaBlock instances
    - Residual connections within each block
    - Layer normalization for training stability
    - Consistent feature dimension throughout
    
    Scaling Characteristics:
    - Time complexity: O(n_blocks * B * L * D * N)
    - Memory complexity: O(n_blocks * B * D * N) for states
    - Linear scaling in number of blocks
    - Memory efficient for long sequences
    
    Apple Silicon Optimization:
    - Each block uses MPS-compatible operations
    - Minimal intermediate tensor allocations
    - Efficient sequential processing
    - Unified memory reduces transfer overhead
    
    Integration Context:
    - Called by MCTModel for RNN-T acoustic modeling
    - Input: Frontend-processed features (B, T/4, D)
    - Output: Encoded representations (B, T/4, D)
    - Compatible with both training and streaming inference
    """
    
    def __init__(self, d_model: int = 256, n_blocks: int = 6, state_dim: int = 16):
        """Initialize Mamba encoder with configurable architecture.
        
        Args:
            d_model: Model dimension for all operations
            n_blocks: Number of Mamba blocks in the stack
            state_dim: State space dimension for each block
        """
        super().__init__()
        
        # Store configuration for reference
        self.d_model = d_model
        self.n_blocks = n_blocks 
        self.state_dim = state_dim
        
        # Create stack of Mamba blocks
        # Each block maintains the same feature dimension
        self.blocks = nn.ModuleList([
            MambaBlock(MambaConfig(d_model=d_model, state_dim=state_dim)) 
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode acoustic sequence through stack of Mamba blocks.
        
        Processes input sequence through all Mamba blocks sequentially,
        allowing each block to refine the representation with its
        selective state space modeling.
        
        Args:
            x: Input sequence features (B, L, D)
               B=batch_size, L=sequence_length, D=d_model
        
        Returns:
            Encoded sequence representations (B, L, D)
            
        Processing Flow:
            x -> block_0 -> block_1 -> ... -> block_{n-1} -> output
            
        Memory Considerations:
        - Each block maintains state tensor (B, D, N)
        - Peak memory during block transitions
        - Residual connections require temporary storage
        """
        # Validate input dimensions
        batch_size, seq_len, feature_dim = x.shape
        assert feature_dim == self.d_model, f"Feature dim {feature_dim} != model dim {self.d_model}"
        
        # Sequential processing through Mamba blocks
        encoded = x
        for block_idx, mamba_block in enumerate(self.blocks):
            # Each block applies: residual(selective_scan(projections(input)))
            encoded = mamba_block(encoded)
            # Shape maintained: (B, L, D)
        
        return encoded
    
    def get_encoder_info(self) -> str:
        """Return encoder configuration and performance information."""
        total_params = sum(p.numel() for p in self.parameters())
        return f"""
        Mamba Encoder Configuration:
        - Blocks: {self.n_blocks}
        - Model dimension: {self.d_model}
        - State dimension: {self.state_dim}
        - Total parameters: {total_params:,}
        - Complexity: O({self.n_blocks} * B * L * {self.d_model} * {self.state_dim})
        - Memory per block: O(B * {self.d_model} * {self.state_dim})
        """
