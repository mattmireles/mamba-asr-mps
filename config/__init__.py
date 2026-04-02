"""
Centralized Configuration Management for MambaASR on Apple Silicon

This module provides centralized access to all configuration constants and
patterns used throughout the MambaASR system. It serves as a single source
of truth for system-wide configuration and enables consistent parameter
management across all components.

Configuration Architecture:
- Hierarchical organization: System > Component > Feature configuration
- Type safety: All constants properly typed and documented  
- Environment integration: Support for environment variable overrides
- Cross-platform compatibility: Apple Silicon optimization with fallbacks
- Documentation: Comprehensive inline documentation for AI-first development

Component Coverage:
- RNN-T Loss: Training constants and backend configuration
- Text Processing: Evaluation and normalization parameters
- Training Pipeline: Hyperparameters and model configuration
- Metrics: Evaluation thresholds and performance targets
- Tokenization: Vocabulary structure and character mapping
- Apple Silicon: MPS optimization and hardware-specific settings

Cross-File Integration:
- Imported by: All training scripts, evaluation tools, and model implementations
- Centralizes: Constants previously scattered across individual modules
- Enables: Consistent configuration management and easy parameter tuning
- Supports: Environment-based configuration override for deployment flexibility

Usage Patterns:
    # Import centralized configuration
    from config import MambaASRConfig
    
    # Access training parameters
    learning_rate = MambaASRConfig.Training.DEFAULT_LEARNING_RATE
    batch_size = MambaASRConfig.Training.DEFAULT_BATCH_SIZE
    
    # Access model configuration
    vocab_size = MambaASRConfig.Model.DEFAULT_VOCAB_SIZE
    d_model = MambaASRConfig.Model.DEFAULT_D_MODEL
    
    # Access evaluation thresholds
    wer_threshold = MambaASRConfig.Metrics.TARGET_WER_PRODUCTION
    cer_threshold = MambaASRConfig.TextProcessing.DEFAULT_CER_THRESHOLD

Environment Override Examples:
    # Override training batch size
    export MAMBA_BATCH_SIZE=4
    
    # Override RNN-T alignment constraint
    export RNNT_MAX_ALIGN=80000
    
    # Override model dimension
    export MAMBA_D_MODEL=512

AI-First Design Principles:
- All constants have comprehensive documentation explaining their purpose
- Cross-component relationships explicitly documented
- Environment variable integration patterns clearly specified
- Hardware-specific optimizations (Apple Silicon) clearly marked
- Performance implications and trade-offs thoroughly explained
"""

# Re-export all configuration classes for convenient access
from .apple_silicon_config import AppleSiliconConfig
from .environment_config import EnvironmentConfig

__all__ = ['AppleSiliconConfig', 'EnvironmentConfig']