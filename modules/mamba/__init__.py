"""
Mamba selective state space models for Apple Silicon speech recognition.

This module provides the core Mamba implementation optimized for Apple Silicon,
including both unidirectional and bidirectional variants for sequence modeling.

Core Components:
- MambaBlock: Single Mamba block with selective state space processing
- MambaConfig: Configuration dataclass for Mamba block parameters
- BiMamba: Bidirectional Mamba for enhanced context modeling
- selective_scan: Core selective scan operation (performance critical)

Apple Silicon Optimizations:
- MPS backend compatibility throughout all operations
- Device-agnostic tensor operations for unified memory
- Numerical stability optimizations for float32 precision
- Profiling annotations for performance analysis

Architecture Integration:
- Used by ConMambaCTC for CTC speech recognition
- Used by MCTModel via MambaEncoder for RNN-T speech recognition
- Core building block for all Mamba-based architectures

Performance Characteristics:
- Linear complexity O(L) in sequence length
- Dominated by selective_scan operation (current bottleneck)
- Memory efficient for long audio sequences
- Suitable for both training and streaming inference

Implementation Status:
- Phase 1: Functional baseline with PyTorch operations
- Phase 2: Custom Metal kernel optimization (future)
- Phase 3: Multi-block fusion optimization (future)

References:
- Mamba paper: Gu & Dao Selective State Space Models
- Selective scan: selective_scan_interface.py
- Apple Silicon guide: README/Mamba-on-Apple-Silicon.md
"""

from .bimamba import BiMamba
from .mamba_blocks import MambaBlock, MambaConfig
from .selective_scan_interface import selective_scan, init_hidden

# Export all public APIs
__all__ = [
    'BiMamba',
    'MambaBlock', 
    'MambaConfig',
    'selective_scan',
    'init_hidden',
]