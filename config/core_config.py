"""
Core Configuration Constants for MambaASR System

This module consolidates all configuration constants from across the MambaASR
system into a single, hierarchically organized structure. It provides the
central configuration authority for all training, evaluation, and deployment
workflows on Apple Silicon.

Configuration Hierarchy:
- MambaASRConfig: Root configuration class
  - Training: Hyperparameters and training-specific constants
  - Model: Architecture and model-specific parameters  
  - RNNTLoss: RNN-T loss computation configuration
  - TextProcessing: Evaluation and text normalization settings
  - Metrics: Performance thresholds and evaluation targets
  - Tokenizer: Vocabulary and character mapping configuration
  - AppleSilicon: Hardware-specific optimization parameters

Design Principles:
1. Single Source of Truth: All constants defined in one location
2. Hierarchical Organization: Logical grouping by component/function
3. Comprehensive Documentation: Every constant explained with context
4. Environment Integration: Support for runtime configuration override
5. Type Safety: All constants properly typed and validated
6. Cross-Reference Documentation: Explicit component relationships

Cross-File Integration:
- Replaces: Scattered constants in individual modules
- Used by: All training scripts, evaluation tools, model implementations
- Enables: Consistent parameter management across entire system
- Supports: Easy configuration tuning and deployment customization

AI-First Documentation Strategy:
- Every constant includes purpose, impact, and usage context
- Cross-component relationships explicitly documented
- Performance implications and trade-offs thoroughly explained
- Environment variable patterns and override mechanisms specified
- Hardware-specific optimizations (Apple Silicon) clearly identified
"""

import os
from typing import Dict, Any


class MambaASRConfig:
    """Centralized configuration for MambaASR system on Apple Silicon.
    
    This class organizes all system configuration into logical hierarchies,
    providing a single source of truth for parameters across training,
    evaluation, and deployment workflows.
    
    Each nested class represents a major system component with its own
    configuration requirements. Constants are organized by functionality
    and documented with their purpose, usage, and performance implications.
    """
    
    class Training:
        """Training pipeline configuration and hyperparameters.
        
        These constants define the training process for Mamba-based speech
        recognition models, optimized for Apple Silicon hardware and unified
        memory architecture.
        """
        
        # Learning Rate Configuration
        DEFAULT_LEARNING_RATE = 3e-4
        """AdamW learning rate optimized for RNN-T training on Apple Silicon.
        
        This rate balances convergence speed with stability for Mamba architectures.
        Lower rates (1e-4) provide more stable training but slower convergence.
        Higher rates (1e-3) may cause instability with RNN-T loss gradients.
        """
        
        MIN_LEARNING_RATE = 1e-6
        """Minimum learning rate for scheduler lower bound."""
        
        MAX_LEARNING_RATE = 1e-2
        """Maximum learning rate to prevent gradient explosion."""
        
        # Batch Size Configuration  
        DEFAULT_BATCH_SIZE = 2
        """Conservative batch size for Apple Silicon + RNN-T memory constraints.
        
        Apple Silicon's unified memory architecture enables larger batches than
        discrete GPU systems, but RNN-T alignment matrices can be memory-intensive.
        Optimal range: 2-8 depending on sequence lengths and model size.
        """
        
        MIN_BATCH_SIZE = 1
        """Minimum batch size for training stability."""
        
        MAX_BATCH_SIZE = 16
        """Maximum recommended batch size for Apple Silicon memory limits."""
        
        # Training Duration
        DEFAULT_EPOCHS = 1
        """Default epoch count for quick testing and validation."""
        
        PRODUCTION_EPOCHS = 100
        """Typical epoch count for production model training."""
        
        # Gradient Management
        GRAD_SET_TO_NONE = True
        """Use set_to_none=True for more efficient memory management."""
        
        DEFAULT_GRAD_CLIP = 1.0
        """Global gradient norm clipping to prevent gradient explosion."""
        
        # Logging and Monitoring
        LOG_INTERVAL = 5
        """Steps between loss logging (more frequent for Mamba training)."""
        
        SYNC_INTERVAL = 1
        """Epochs between device synchronization for accurate timing."""
        
        # Checkpoint Management
        CHECKPOINT_INTERVAL = 10
        """Epochs between checkpoint saves."""
        
        MAX_CHECKPOINTS = 5
        """Maximum number of checkpoints to retain."""
    
    
    class Model:
        """Model architecture configuration for Mamba-based speech recognition.
        
        These parameters define the MCT (Mamba-CNN-Transducer) architecture
        optimized for Apple Silicon deployment and streaming inference.
        """
        
        # Core Architecture
        DEFAULT_D_MODEL = 256
        """Model dimension for Mamba encoder and other components.
        
        This dimension balances model capacity with computational efficiency.
        Common values: 256 (efficient), 512 (standard), 1024 (large)
        Apple Silicon optimization favors 256 for memory bandwidth efficiency.
        """
        
        DEFAULT_N_BLOCKS = 4
        """Number of Mamba blocks in encoder (lighter for RNN-T complexity).
        
        Fewer blocks reduce computational overhead while maintaining the
        sequence modeling capabilities that make Mamba effective for speech.
        """
        
        DEFAULT_JOINT_DIM = 320
        """RNN-T joiner dimension for acoustic-linguistic fusion.
        
        The joiner network combines encoder and predictor representations.
        Should be >= d_model for sufficient representational capacity.
        """
        
        DEFAULT_VOCAB_SIZE = 1024
        """Default vocabulary size (may be overridden for character tokenizer).
        
        Large vocab supports subword tokenization if needed, but character
        tokenizer uses 29 tokens for memory efficiency on Apple Silicon.
        """
        
        # Memory Management
        MAX_SEQUENCE_LENGTH = 10000
        """Maximum sequence length for memory management."""
        
        # Frontend Configuration
        MEL_FEATURES = 80
        """Mel-spectrogram feature dimension (standard for speech recognition)."""
        
        SAMPLE_RATE = 16000
        """Audio sample rate in Hz (standard for speech recognition)."""
    
    
    class RNNTLoss:
        """RNN-T loss computation configuration for Apple Silicon.
        
        These constants control RNN-T loss behavior, memory management,
        and backend selection for optimal performance on Apple Silicon.
        """
        
        # Memory Management Constants
        DEFAULT_MAX_ALIGNMENT = 60000
        """Default maximum alignment constraint for T*U dimension product.
        
        Prevents memory pressure on Apple Silicon by capping alignment grid size.
        Based on empirical testing showing stable performance below this threshold.
        Environment override: RNNT_MAX_ALIGN
        """
        
        MIN_ALIGNMENT_THRESHOLD = 1
        """Minimum alignment value to ensure numerical stability."""
        
        DEFAULT_CLAMP_VALUE = -1
        """Default clamp value for RNN-T loss (-1 disables clamping)."""
        
        # Backend Selection
        MAX_BACKEND_ATTEMPTS = 3
        """Maximum backend selection attempts before fallback."""
        
        DEFAULT_REDUCTION = "mean"
        """Default reduction mode for loss aggregation across batch."""
        
        # RNN-T Specific
        RNNT_BLANK_TOKEN = 0
        """RNN-T blank token index for alignment."""
        
        MAX_ALIGNMENT_SIZE = 1000000
        """Maximum T*U for memory safety."""
        
        # Performance Tuning
        NAIVE_RNN_T_MAX_TIME = 64
        """Maximum T frames for naive RNN-T implementation."""
        
        NAIVE_RNN_T_MAX_TOKENS = 16
        """Maximum U tokens for naive RNN-T implementation."""
    
    
    class TextProcessing:
        """Text processing and evaluation configuration.
        
        Constants for text normalization, error rate computation, and
        evaluation workflows optimized for speech recognition assessment.
        """
        
        # Character Set for Normalization
        ALLOWED_CHARACTERS = set("abcdefghijklmnopqrstuvwxyz0123456789'")
        """Allowed characters for text normalization (lowercase + digits + apostrophe)."""
        
        # Exit Codes for CI/CD Integration
        EXIT_SUCCESS = 0
        """Normal completion, all thresholds met."""
        
        EXIT_MISSING_TRANSCRIPTS = 2
        """Missing/empty transcripts with --strict enabled."""
        
        EXIT_THRESHOLD_EXCEEDED = 3
        """Error rates exceed specified thresholds."""
        
        # File Processing
        DEFAULT_TRANSCRIPT_PATTERN = "transcript_*_*.txt"
        """Default glob pattern for transcript file discovery."""
        
        MAX_TRANSCRIPT_LINE_LENGTH = 10000
        """Maximum line length for transcript extraction."""
        
        # Error Rate Configuration
        ERROR_RATE_PRECISION = 3
        """Decimal places for error rate reporting."""
        
        MIN_SEQUENCE_LENGTH = 1
        """Minimum sequence length to avoid division by zero."""
        
        # Evaluation Thresholds
        DEFAULT_CER_THRESHOLD = 0.6
        """Default Character Error Rate threshold for evaluation gating."""
        
        DEFAULT_WER_THRESHOLD = 0.3
        """Default Word Error Rate threshold for evaluation gating."""
    
    
    class Metrics:
        """Performance metrics and evaluation thresholds.
        
        These constants define success criteria and performance targets
        for speech recognition evaluation across different deployment scenarios.
        """
        
        # WER Thresholds
        EXCELLENT_WER = 0.05
        """WER below 5% considered excellent (commercial quality)."""
        
        GOOD_WER = 0.10
        """WER below 10% considered good (usable for most applications)."""
        
        ACCEPTABLE_WER = 0.20
        """WER below 20% considered acceptable (may need improvement)."""
        
        POOR_WER = 0.50
        """WER above 50% considered poor (significant issues)."""
        
        # Performance Targets
        TARGET_WER_LIBRISPEECH = 0.03
        """Target WER for LibriSpeech test-clean benchmark."""
        
        TARGET_WER_PRODUCTION = 0.10
        """Acceptable WER for production deployment."""
        
        # Algorithm Parameters
        MAX_SEQUENCE_LENGTH = 10000
        """Maximum sequence length for DP optimization."""
        
        @staticmethod
        def get_wer_interpretation(wer: float) -> str:
            """Return interpretation of WER value with emoji indicators."""
            if wer <= MambaASRConfig.Metrics.EXCELLENT_WER:
                return "🟢 Excellent performance"
            elif wer <= MambaASRConfig.Metrics.GOOD_WER:
                return "🟡 Good performance"
            elif wer <= MambaASRConfig.Metrics.ACCEPTABLE_WER:
                return "🟠 Acceptable performance"
            elif wer <= MambaASRConfig.Metrics.POOR_WER:
                return "🔴 Poor performance"
            else:
                return "💥 Critical performance issues"
    
    
    class Tokenizer:
        """Character tokenizer configuration for speech recognition.
        
        These constants define the vocabulary structure and token mappings
        for character-level speech recognition on Apple Silicon hardware.
        """
        
        # Vocabulary Structure
        BLANK_TOKEN_ID = 0
        """CTC/RNN-T blank token index."""
        
        VOCABULARY_SIZE = 29
        """Total tokens: blank + space + a-z + apostrophe."""
        
        # Character Set
        ALPHABET_SIZE = 26
        """English alphabet a-z."""
        
        SPECIAL_CHAR_COUNT = 2
        """Space and apostrophe."""
        
        # Token Mapping
        SPACE_TOKEN_ID = 1
        """Space character token."""
        
        FIRST_LETTER_ID = 2
        """Start of alphabet range (a)."""
        
        LAST_LETTER_ID = 27
        """End of alphabet range (z)."""
        
        APOSTROPHE_TOKEN_ID = 28
        """Apostrophe for contractions."""
        
        @staticmethod
        def get_vocabulary_info() -> str:
            """Return vocabulary structure documentation."""
            return f"""
            Character Tokenizer Vocabulary:
            - Index 0: Blank token (CTC/RNN-T alignment)
            - Index 1: Space character (word boundaries)
            - Index 2-27: Lowercase letters a-z
            - Index 28: Apostrophe (contractions like don't, can't)
            - Total size: {MambaASRConfig.Tokenizer.VOCABULARY_SIZE} tokens
            - Memory efficient: Minimal embedding table size
            """
    
    
    @classmethod
    def get_environment_overrides(cls) -> Dict[str, Any]:
        """Get current environment variable overrides for configuration.
        
        Returns:
            Dictionary mapping configuration paths to environment values
        """
        overrides = {}
        
        # Training overrides
        if batch_size := os.getenv('MAMBA_BATCH_SIZE'):
            overrides['Training.DEFAULT_BATCH_SIZE'] = int(batch_size)
            
        if learning_rate := os.getenv('MAMBA_LEARNING_RATE'):
            overrides['Training.DEFAULT_LEARNING_RATE'] = float(learning_rate)
        
        # Model overrides
        if d_model := os.getenv('MAMBA_D_MODEL'):
            overrides['Model.DEFAULT_D_MODEL'] = int(d_model)
            
        if vocab_size := os.getenv('MAMBA_VOCAB_SIZE'):
            overrides['Model.DEFAULT_VOCAB_SIZE'] = int(vocab_size)
        
        # RNN-T Loss overrides
        if max_align := os.getenv('RNNT_MAX_ALIGN'):
            overrides['RNNTLoss.DEFAULT_MAX_ALIGNMENT'] = int(max_align)
        
        return overrides
    
    
    @classmethod
    def get_configuration_summary(cls) -> str:
        """Generate comprehensive configuration summary for logging/debugging.
        
        Returns:
            Formatted string with all current configuration values
        """
        env_overrides = cls.get_environment_overrides()
        override_info = f"Environment Overrides: {env_overrides}" if env_overrides else "No environment overrides"
        
        return f"""
        MambaASR Configuration Summary:
        
        Training:
        - Learning Rate: {cls.Training.DEFAULT_LEARNING_RATE}
        - Batch Size: {cls.Training.DEFAULT_BATCH_SIZE}
        - Epochs: {cls.Training.DEFAULT_EPOCHS}
        - Gradient Clipping: {cls.Training.DEFAULT_GRAD_CLIP}
        
        Model:
        - Model Dimension: {cls.Model.DEFAULT_D_MODEL}
        - Mamba Blocks: {cls.Model.DEFAULT_N_BLOCKS}
        - Joint Dimension: {cls.Model.DEFAULT_JOINT_DIM}
        - Vocabulary Size: {cls.Model.DEFAULT_VOCAB_SIZE}
        
        RNN-T Loss:
        - Max Alignment: {cls.RNNTLoss.DEFAULT_MAX_ALIGNMENT}
        - Blank Token: {cls.RNNTLoss.RNNT_BLANK_TOKEN}
        - Reduction: {cls.RNNTLoss.DEFAULT_REDUCTION}
        
        Metrics:
        - Target Production WER: {cls.Metrics.TARGET_WER_PRODUCTION}
        - Excellent WER Threshold: {cls.Metrics.EXCELLENT_WER}
        
        Tokenizer:
        - Vocabulary Size: {cls.Tokenizer.VOCABULARY_SIZE}
        - Blank Token ID: {cls.Tokenizer.BLANK_TOKEN_ID}
        
        {override_info}
        """