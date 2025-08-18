"""
Mamba-ASR-MPS module exports for Apple Silicon speech recognition.

This module provides the main entry points for the Mamba-ASR-MPS package,
exporting the primary model architectures and configurations optimized
for Apple Silicon deployment.

Exported Models:
- ConMambaCTC: CTC-based speech recognition with Mamba encoder
- MCTModel: RNN-T speech recognition with Mamba encoder

Exported Configurations:
- ConMambaCTCConfig: Configuration for CTC training pipeline
- MCTConfig: Configuration for RNN-T training pipeline

Architectural Context:
- Both models use Mamba selective state space models for sequence processing
- Optimized for Apple Silicon MPS backend with comprehensive fallback support
- Designed for both training and inference on unified memory architecture

Usage Examples:
    # CTC-based speech recognition
    from modules import ConMambaCTC, ConMambaCTCConfig
    config = ConMambaCTCConfig(d_model=256, n_blocks=4)
    model = ConMambaCTC(config)
    
    # RNN-T speech recognition
    from modules import MCTModel, MCTConfig
    config = MCTConfig(d_model=256, n_blocks=6)
    model = MCTModel(config)

Integration Points:
- Used by train_CTC.py for CTC training pipeline
- Used by train_RNNT.py for RNN-T training pipeline
- Imported by inference and evaluation scripts

Apple Silicon Features:
- MPS backend compatibility throughout
- Unified memory architecture optimization
- Mixed precision training support
- Comprehensive profiling integration

References:
- ConMamba architecture: modules/Conmamba.py
- MCT architecture: modules/mct/mct_model.py
- Mamba blocks: modules/mamba/mamba_blocks.py
- Training guides: README/Mamba-on-Apple-Silicon.md
"""

from .Conmamba import ConMambaCTC, ConMambaCTCConfig
from .mct.mct_model import MCTModel, MCTConfig

# Optional RNNT MPS-native loss facade
try:  # pragma: no cover
    from .rnnt_loss_mps import rnnt_loss_mps  # type: ignore
except Exception:
    rnnt_loss_mps = None  # type: ignore

# Export all public APIs
__all__ = [
    'ConMambaCTC',
    'ConMambaCTCConfig', 
    'MCTModel',
    'MCTConfig',
]

# Version and package information
__version__ = '1.0.0-alpha'
__description__ = 'Mamba-based speech recognition optimized for Apple Silicon'
__author__ = 'Apple Silicon AI Research Team'
