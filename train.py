#!/usr/bin/env python3
"""
End-to-end training script for Mamba-ASR with a learned 1024→29 projection head.

This script trains a production-ready projection head on top of a 1024-class
ConMamba CTC backbone and automates validation, checkpointing, and post-run
export/evaluation. It serves as the bridge between the PyTorch training pipeline
and the CoreML deployment system used by Swift applications.

Key responsibilities:
- Training learned projection head for character-level CTC loss
- Comprehensive validation with Character Error Rate (CER) metrics
- Automatic checkpointing and best model selection
- Integration with CoreML export pipeline for production deployment
- Apple Silicon MPS optimization with CPU fallback for unsupported operations

Called by:
- Direct command-line execution for standalone Mamba-ASR training
- Batch training scripts for hyperparameter sweeps and experiments
- CI/CD pipelines for automated model training and validation
- Development workflows for iterative model improvement
- Research experiments comparing Mamba vs Transformer architectures

Calls to:
- datasets/librispeech_csv.py:LibriSpeechCSVDataset for efficient CSV-based data loading
- modules/Conmamba.py:ConMambaCTC for the core Mamba encoder backbone
- utils/tokenizer.py:CharTokenizer for character-level tokenization and normalization
- utils/hardware.py:get_optimal_worker_count() for Apple Silicon worker optimization
- scripts/extract_projection_from_ckpt.py for learned projection weight extraction
- scripts/eval_batch.sh for comprehensive CoreML evaluation harness
- External: torch.nn.CTCLoss, torch.optim.AdamW, torch.optim.lr_scheduler.CosineAnnealingLR

System architecture integration:
- Backbone: ConMambaCTC with configurable vocab_size=1024 to produce per-frame logits
- Projection Head: Final torch.nn.Linear(1024, 29) named `proj` for character vocab
- Loss Function: CTC loss over 29-character vocabulary (blank token at index 0)
- Validation: CTC loss + Character Error Rate via greedy CTC decoding
- Checkpointing: Regular snapshots with best model selection based on lowest validation CER
- Post-training: Automatic projection extraction and CoreML evaluation harness execution

Design rationale:
- The CoreML runtime and Swift runner already operate with a 1024-wide vocabulary internally
- Learning a 1024→29 projection on real speech data provides accurate, data-driven character mappings
- The learned projection weights are exported to exports/projection_1024x29.csv for CoreML integration
- This approach enables efficient character-level recognition while maintaining CoreML compatibility

Apple Silicon optimizations:
- MPS acceleration leverages unified memory architecture for efficient training
- CPU fallback enabled for CTC loss computation (PYTORCH_ENABLE_MPS_FALLBACK=1)
- Unified memory pressure management prevents system-wide swapping
- Memory synchronization patterns optimized for Apple Silicon architecture
- Worker count auto-detection considers Apple Silicon multi-core performance characteristics

Training pipeline integration:
- Input: CSV manifests with audio paths, durations, and transcriptions
- Processing: Mel-spectrogram feature extraction and character tokenization
- Training: CTC loss optimization with AdamW and cosine annealing schedule
- Validation: Real-time CER computation and best model checkpoint selection
- Output: Production-ready model checkpoints and CoreML-compatible projection weights

Usage examples:
    # Minimal sanity check on a tiny CSV for development
    python train.py \
        --train-csv /path/to/train.csv \
        --val-csv /path/to/val.csv \
        --epochs 2 --batch-size 2

    # Full production training with optimal hyperparameters
    python train.py \
        --train-csv data/train.csv --val-csv data/val.csv \
        --epochs 30 --batch-size 8 --lr 3e-4 --d-model 256 --n-blocks 6

    # Freeze backbone training (projection head only) for faster fine-tuning
    python train.py --freeze-backbone --epochs 10

Output artifacts:
- Training checkpoints: exports/checkpoints/
- Best model checkpoint: best.pt (selected by lowest validation CER)
- Learned projection weights: exports/projection_1024x29.csv
- Evaluation reports: exports/CoreMLTraces/wer_cer_overview_opt.md
- Training logs: Comprehensive progress tracking and performance metrics
"""
from __future__ import annotations

import os
import math
import time
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

# Enable CPU fallback for missing MPS ops (e.g., aten::_ctc_loss)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
from pathlib import Path as _PathAdd
# Ensure local package imports work regardless of invocation path
_here = _PathAdd(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from datasets.librispeech_csv import LibriSpeechCSVDataset, DatasetConstants as DS
import importlib.util as _ilu
def _load_char_tokenizer() -> type:
    tok_path = _here / "utils" / "tokenizer.py"
    spec = _ilu.spec_from_file_location("mambautils_tokenizer", str(tok_path))
    assert spec and spec.loader
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return getattr(mod, "CharTokenizer")

CharTokenizer = _load_char_tokenizer()
from modules.Conmamba import ConMambaCTC, ConMambaCTCConfig


class MambaTrainingConstants:
    """
    Named constants for Mamba-ASR training configuration and model architecture.
    
    This class centralizes all training-related constants to eliminate magic numbers
    throughout the training pipeline. Constants are organized by category and include
    detailed documentation explaining their purpose and relationship to the overall
    system architecture.
    
    Used throughout:
    - Model architecture definition for vocabulary sizes and dimensions
    - Training hyperparameter configuration and optimization settings
    - Apple Silicon MPS optimization and memory management settings
    - CTC decoding and character tokenization constants
    - Performance monitoring and logging configuration
    - Checkpoint management and validation scheduling
    """
    
    # MARK: - Model Architecture Constants
    
    # Vocabulary size constants defining the model's token output spaces
    # These sizes are critical for CoreML compatibility and deployment integration
    BACKBONE_VOCAB_SIZE = 1024          # ConMamba CTC backbone output vocabulary size
    CHARACTER_VOCAB_SIZE = 29           # Final character vocabulary size (a-z, space, apostrophe, blank)
    CTC_BLANK_TOKEN_ID = 0              # CTC blank token identifier (standard CTC convention)
    PROJECTION_INPUT_DIM = 1024         # Input dimension for learned projection head
    PROJECTION_OUTPUT_DIM = 29          # Output dimension for learned projection head
    
    # Model dimension constants for ConMamba architecture
    # These defaults provide good balance between accuracy and computational efficiency
    DEFAULT_D_MODEL = 256               # Default hidden dimension for Mamba blocks
    DEFAULT_N_BLOCKS = 6                # Default number of Mamba encoder blocks
    
    # MARK: - Training Hyperparameter Constants
    
    # Learning rate and optimization constants optimized for Mamba architecture
    DEFAULT_LEARNING_RATE = 3e-4        # AdamW learning rate for stable Mamba training
    DEFAULT_WEIGHT_DECAY = 1e-2         # L2 regularization strength for generalization
    DEFAULT_GRADIENT_CLIP = 5.0         # Gradient clipping threshold for training stability
    
    # Batch size constants optimized for Apple Silicon memory architecture
    DEFAULT_BATCH_SIZE = 4              # Default batch size balancing memory usage and convergence
    MINIMUM_BATCH_SIZE = 1              # Minimum viable batch size for memory-constrained systems
    
    # Training schedule constants for effective learning
    DEFAULT_EPOCHS = 10                 # Default number of training epochs for convergence
    DEFAULT_EVAL_EVERY_EPOCHS = 1       # Validation frequency (every N epochs)
    MINIMUM_EVAL_EPOCHS = 1             # Minimum validation frequency to prevent overfitting
    
    # MARK: - Apple Silicon Optimization Constants
    
    # DataLoader worker constants optimized for Apple Silicon performance
    DEFAULT_NUM_WORKERS = 0             # Apple Silicon typically performs better with single-threaded I/O
    AUTO_DETECT_WORKERS = -1            # Flag indicating automatic worker count detection
    FALLBACK_WORKER_COUNT = 0           # Conservative fallback when auto-detection fails
    
    # Memory management constants for Apple Silicon unified memory architecture
    DEFAULT_PIN_MEMORY = False          # Disable memory pinning for unified memory systems
    MPS_MEMORY_SYNC_REQUIRED = True     # MPS requires explicit synchronization for accurate timing
    
    # MARK: - Performance Monitoring Constants
    
    # Logging and monitoring intervals for training progress tracking
    DEFAULT_LOG_INTERVAL = 25           # Training step interval for loss logging
    PERFORMANCE_LOG_EVERY = 100         # Default interval for performance monitoring logs
    MINIMUM_PERFORMANCE_LOG_INTERVAL = 50  # Minimum performance logging frequency
    
    # Performance monitoring thresholds and display precision
    PERCENTAGE_SCALE_FACTOR = 100.0     # Convert ratios to percentages for display
    LOSS_DISPLAY_PRECISION = 4          # Decimal places for loss value display
    TIME_DISPLAY_PRECISION = 1          # Decimal places for timing displays
    CER_DISPLAY_PRECISION = 4           # Decimal places for CER metric display
    
    # MARK: - Checkpointing and Model Persistence Constants
    
    # Checkpoint file naming conventions for model persistence
    LAST_CHECKPOINT_NAME = "last.pt"    # Filename for most recent checkpoint
    BEST_CHECKPOINT_NAME = "best.pt"    # Filename for best validation performance checkpoint
    DEFAULT_CHECKPOINT_DIR = "exports/checkpoints"  # Default checkpoint directory
    
    # Model export constants for CoreML integration
    PROJECTION_WEIGHT_KEY = "proj.weight"   # State dict key for projection layer weights
    PROJECTION_BIAS_KEY = "proj.bias"       # State dict key for projection layer bias
    PROJECTION_CSV_FILENAME = "projection_1024x29.csv"  # Exported projection filename
    
    # MARK: - Data Processing Constants
    
    # CTC processing constants for loss computation and decoding
    CTC_ZERO_INFINITY = True            # Enable zero_infinity for CTC loss stability
    COSINE_SCHEDULER_T_MAX_FALLBACK = 1 # Minimum T_max for cosine annealing scheduler
    
    # Validation and error handling constants
    MINIMUM_LOSS_BATCHES = 1            # Minimum batches required for loss averaging
    MINIMUM_TOTAL_TIME = 0.0            # Minimum time threshold for performance calculations
    
    # Text processing constants for character-level tokenization
    BLANK_CHAR_TOKEN_ID = 0             # Character tokenizer blank token identifier
    SPACE_TOKEN_ID = 1                  # Character tokenizer space token identifier (typical mapping)
    
    # MARK: - Post-training Integration Constants
    
    # External script integration constants for automated workflows
    PYTHON3_EXECUTABLE = "python3"      # Python interpreter for script execution
    BASH_EXECUTABLE = "bash"            # Shell interpreter for batch script execution
    SUCCESSFUL_EXIT_CODE = 0            # Unix success exit code
    
    # File system constants for post-training artifact management
    PROJECTION_EXPORT_SUBDIR = "exports"           # Subdirectory for projection exports
    COREML_TRACES_SUBDIR = "exports/CoreMLTraces"  # Subdirectory for evaluation traces
    EVAL_SCRIPT_SUBDIR = "scripts"                 # Subdirectory for evaluation scripts
    
    # External script filenames for automated post-training workflows
    PROJECTION_EXTRACTOR_SCRIPT = "extract_projection_from_ckpt.py"  # Projection extraction script
    BATCH_EVAL_SCRIPT = "eval_batch.sh"                             # CoreML evaluation harness script


# -----------------------------
# Utility: device selection (MPS → CUDA → CPU)
# -----------------------------
def get_device() -> torch.device:
    """
    Selects optimal compute device following Apple Silicon > CUDA > CPU priority hierarchy.
    
    This function implements the core device selection logic for Mamba-ASR training,
    prioritizing Apple Silicon MPS when available for unified memory architecture benefits,
    falling back to CUDA for discrete GPU acceleration, and finally CPU as universal fallback.
    
    Called by:
    - main() function during training initialization for model and tensor placement
    - Training loop setup for device-specific optimization configuration
    - Model loading and checkpoint management for consistent device placement
    
    Device selection priority and rationale:
    1. Apple Silicon MPS: Leverages unified memory architecture and Apple Neural Engine
       - Optimal for Apple Silicon M1/M2/M3 systems with integrated GPU acceleration
       - Benefits from zero-copy memory access between CPU and GPU operations
       - Requires CPU fallback for unsupported operations (CTC loss computation)
    2. NVIDIA CUDA: High-performance discrete GPU acceleration
       - Optimal for systems with dedicated NVIDIA GPUs and CUDA drivers
       - Provides full operation support without CPU fallback requirements
       - Enables larger batch sizes through discrete GPU memory
    3. CPU fallback: Universal compatibility across all hardware platforms
       - Ensures training capability on any system regardless of GPU availability
       - Slower training speed but guaranteed compatibility and stability
    
    Apple Silicon considerations:
    - MPS backend requires PYTORCH_ENABLE_MPS_FALLBACK=1 for CTC loss compatibility
    - Unified memory architecture eliminates traditional GPU memory limitations
    - Memory pressure management critical to prevent system-wide swapping
    
    Returns:
        torch.device: Selected compute device optimized for current hardware configuration
        
    Example usage:
        device = get_device()  # Returns torch.device("mps") on Apple Silicon M1/M2/M3
        model = model.to(device)  # Place model on optimal device
        tensors = tensors.to(device)  # Ensure tensor-model device consistency
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------
# Data collation for CTC
# -----------------------------
def ctc_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]):
    """
    Collates LibriSpeechCSVDataset samples into CTC-compatible batch tensors.
    
    This function transforms individual dataset samples into properly padded and structured
    batch tensors required for CTC loss computation and model training. It handles variable-length
    audio features and token sequences while maintaining efficiency for Apple Silicon training.
    
    Called by:
    - PyTorch DataLoader during training batch construction in main() training loop
    - PyTorch DataLoader during validation batch construction in run_validation()
    - Multiprocessing workers when num_workers > 0 for parallel batch preparation
    
    Input processing pipeline:
    - Individual samples: (mel_features[T,80], feature_length, tokens[U], text_string)
    - Batch assembly: Pad variable-length sequences to maximum length within batch
    - Memory optimization: Use contiguous tensors for efficient GPU memory access
    - Type consistency: Ensure appropriate tensor dtypes for CTC loss requirements
    
    CTC-specific collation requirements:
    - Features: Zero-padded to max_T within batch for parallel processing efficiency
    - Feature lengths: Exact sequence lengths for CTC input length specification
    - Targets: Concatenated token sequences for CTC target format requirement
    - Target lengths: Per-utterance token counts for CTC target length specification
    
    Apple Silicon optimizations:
    - Contiguous memory layout leverages unified memory architecture efficiently
    - Float32 dtype ensures MPS backend compatibility without precision issues
    - Long dtype for indices ensures compatibility with CTC loss implementation
    
    Args:
        batch: List of dataset samples, each containing:
            - mel_features (torch.Tensor): Mel-spectrogram features [time_steps, n_mels]
            - feature_length (torch.Tensor): Actual length of audio features
            - tokens (torch.Tensor): Character token sequence [sequence_length]  
            - text (str): Original transcription text for reference and debugging
            
    Returns:
        Tuple containing CTC-ready batch tensors:
        - feats (torch.Tensor): Padded mel features [batch_size, max_time, n_mels]
        - feat_lens (torch.Tensor): Actual feature lengths [batch_size]
        - targets (torch.Tensor): Concatenated token sequences [total_tokens]
        - target_lens (torch.Tensor): Per-sample token counts [batch_size]
        - texts (List[str]): Original transcription texts for validation and logging
        
    Memory considerations:
    - Padding increases memory usage proportional to longest sequence in batch
    - Concatenated targets require total token count across entire batch
    - Zero-padding enables vectorized computation at cost of memory efficiency
    """
    feats_list, feat_lens, tokens_list, texts = zip(*batch)
    batch_size = len(batch)

    # Find maximum time dimension for padding all features to consistent shape
    max_time_steps = max(features.shape[0] for features in feats_list)
    
    # Create zero-padded feature tensor with shape [batch_size, max_time, n_mels]
    # Use DS.N_MELS constant from dataset configuration for mel-spectrogram dimensions
    padded_feats = torch.zeros(batch_size, max_time_steps, DS.N_MELS, dtype=torch.float32)
    for sample_idx, features in enumerate(feats_list):
        # Copy actual features into padded tensor, leaving remainder as zeros
        padded_feats[sample_idx, :features.shape[0]] = features
    
    # Stack feature lengths into batch tensor for CTC input length specification
    feat_lens_tensor = torch.stack(list(feat_lens)).to(torch.long)

    # Concatenate all token sequences into single tensor for CTC target format
    # CTC loss expects targets as concatenated sequences rather than padded batch
    concatenated_targets = torch.cat(list(tokens_list)).to(torch.long)
    
    # Create tensor of per-sample target lengths for CTC target length specification
    target_lens_tensor = torch.tensor([tokens.shape[0] for tokens in tokens_list], dtype=torch.long)

    return padded_feats, feat_lens_tensor, concatenated_targets, target_lens_tensor, list(texts)


# -----------------------------
# Model wrapper with learned 1024→29 head
# -----------------------------
class MambaASRForCTC(nn.Module):
    """
    ConMamba backbone with learned projection head for character-level CTC training.
    
    This model combines a ConMambaCTC encoder backbone with a learned linear projection
    head to bridge between the backbone's large vocabulary space and the final character
    vocabulary required for speech recognition. The design enables CoreML compatibility
    while maintaining training flexibility.
    
    Architecture components:
    - Backbone: ConMambaCTC encoder configured for large intermediate vocabulary
    - Projection Head: Linear layer mapping from backbone output to character vocabulary
    - Integration: End-to-end training with CTC loss for character sequence prediction
    
    Used by:
    - main() training loop for model instantiation and forward pass computation
    - run_validation() function for validation loss and CER metric computation
    - Checkpoint saving and loading operations for model persistence
    - Post-training projection weight extraction for CoreML deployment
    
    Calls to:
    - modules.Conmamba:ConMambaCTC for core Mamba encoder backbone processing
    - modules.Conmamba:ConMambaCTCConfig for backbone configuration setup
    - torch.nn.Linear for learned projection head implementation
    
    Design rationale:
    - Large backbone vocabulary (1024) provides rich intermediate representations
    - Learned projection enables data-driven character vocabulary mapping
    - Named projection layer ('proj') enables easy weight extraction for CoreML
    - Bias term in projection provides learned character frequency adaptation
    
    CoreML integration considerations:
    - Projection weights exported as CSV for Swift application integration
    - Backbone vocabulary size matches CoreML model expectations (1024 classes)
    - Character vocabulary size matches tokenizer configuration (29 classes)
    - Layer naming convention enables automated weight extraction scripts
    
    Training workflow:
    1. Forward pass: mel_features → backbone → projection → character_logits
    2. Loss computation: CTC loss over character vocabulary predictions
    3. Validation: Greedy CTC decoding for Character Error Rate calculation
    4. Checkpointing: Complete model state preservation for resumption/deployment
    """

    def __init__(self, d_model: int = MambaTrainingConstants.DEFAULT_D_MODEL, 
                 n_blocks: int = MambaTrainingConstants.DEFAULT_N_BLOCKS):
        """
        Initialize MambaASR model with ConMamba backbone and projection head.
        
        Args:
            d_model: Hidden dimension for Mamba blocks (default optimized for Apple Silicon)
            n_blocks: Number of Mamba encoder blocks (default balanced for accuracy/speed)
        """
        super().__init__()
        
        # Configure ConMamba backbone with large intermediate vocabulary
        # Uses BACKBONE_VOCAB_SIZE for rich intermediate representations
        backbone_config = ConMambaCTCConfig(
            d_model=d_model, 
            n_blocks=n_blocks, 
            vocab_size=MambaTrainingConstants.BACKBONE_VOCAB_SIZE
        )
        self.backbone = ConMambaCTC(backbone_config)
        
        # Learned projection head from backbone vocabulary to character vocabulary
        # Named 'proj' for easy extraction by scripts/extract_projection_from_ckpt.py
        self.proj = nn.Linear(
            MambaTrainingConstants.PROJECTION_INPUT_DIM,
            MambaTrainingConstants.PROJECTION_OUTPUT_DIM,
            bias=True
        )

    def forward(self, feats: torch.Tensor, feat_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass from mel-spectrogram features to character-level logits.
        
        Processing pipeline:
        1. Backbone processing: mel_features → ConMamba → backbone_logits[1024]
        2. Projection mapping: backbone_logits → learned_projection → character_logits[29]
        3. Length preservation: Maintain sequence length information for CTC loss
        
        Args:
            feats: Mel-spectrogram features [batch_size, time_steps, n_mels]
            feat_lens: Actual feature lengths [batch_size] for sequence masking
            
        Returns:
            Tuple containing:
            - character_logits: Logits over character vocabulary [batch_size, time_steps', 29]
            - output_lengths: Sequence lengths after backbone subsampling [batch_size]
        """
        # Process through ConMamba backbone to get intermediate representations
        backbone_logits, output_lengths = self.backbone(feats, feat_lens)  # [B, T', 1024]
        
        # Apply learned projection to map to character vocabulary space
        character_logits = self.proj(backbone_logits)  # [B, T', 29]
        
        return character_logits, output_lengths


# -----------------------------
# Decoding and metrics (CER)
# -----------------------------
@dataclass
class CERScore:
    total_chars: int = 0
    total_errors: int = 0

    def update(self, ref: str, hyp: str) -> None:
        r = list(ref.replace(" ", ""))
        h = list(hyp.replace(" ", ""))
        if len(r) == 0:
            # If reference is empty, count all hyp chars as errors
            self.total_errors += len(h)
            self.total_chars += max(len(h), 1)
            return
        # Levenshtein distance
        la, lb = len(r), len(h)
        dp = [[0] * (lb + 1) for _ in range(la + 1)]
        for i in range(la + 1):
            dp[i][0] = i
        for j in range(lb + 1):
            dp[0][j] = j
        for i in range(1, la + 1):
            for j in range(1, lb + 1):
                cost = 0 if r[i - 1] == h[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        self.total_errors += dp[la][lb]
        self.total_chars += len(r)

    @property
    def cer(self) -> float:
        return (self.total_errors / self.total_chars) if self.total_chars > 0 else 0.0


def ctc_greedy_decode(logits_29: torch.Tensor, blank_id: int = MambaTrainingConstants.CTC_BLANK_TOKEN_ID) -> List[List[int]]:
    """
    Performs greedy CTC decoding on per-frame character logits for validation evaluation.
    
    This function implements standard CTC greedy decoding by selecting the most probable
    character at each time step, removing blank tokens, and collapsing consecutive repeats.
    It's used during validation to compute Character Error Rate (CER) metrics.
    
    Called by:
    - run_validation() during validation loop for CER computation
    - Training loop validation phases for real-time accuracy monitoring
    - Post-training evaluation scripts requiring character sequence predictions
    
    CTC decoding algorithm:
    1. Argmax selection: Choose most probable character at each time step
    2. Blank removal: Filter out CTC blank tokens (ID=0)  
    3. Repeat collapse: Merge consecutive identical characters
    4. Sequence assembly: Build final character token sequences
    
    Args:
        logits_29: Character logits tensor [batch_size, time_steps, 29]
                   Output from MambaASRForCTC model projection head
        blank_id: CTC blank token identifier (uses named constant for consistency)
        
    Returns:
        List of decoded token ID sequences, one per batch sample
        Each sequence contains character token IDs without blanks or repeats
        
    Example:
        Input logits: [batch_size=2, time_steps=10, vocab_size=29]  
        Output: [[5, 8, 12, 12, 15], [1, 14, 7]]  # Character token sequences
    """
    with torch.no_grad():
        # Select most probable character at each time step
        predicted_tokens = logits_29.argmax(dim=-1)  # [batch_size, time_steps]
    
    decoded_sequences: List[List[int]] = []
    
    # Process each sequence in the batch
    for token_sequence in predicted_tokens:
        previous_token = blank_id
        decoded_sequence: List[int] = []
        
        # Apply CTC decoding rules: remove blanks and collapse repeats
        for current_token in token_sequence.tolist():
            if current_token != previous_token and current_token != blank_id:
                decoded_sequence.append(current_token)
            previous_token = current_token
            
        decoded_sequences.append(decoded_sequence)
    
    return decoded_sequences


def ids_to_text(token_ids: List[int], tokenizer: CharTokenizer) -> str:
    """
    Converts character token IDs to text string using CharTokenizer mapping.
    
    This function translates decoded CTC token sequences back into human-readable text
    for validation, logging, and evaluation purposes. It handles the character tokenizer's
    specific ID-to-character mapping and filters out any invalid or blank tokens.
    
    Called by:
    - run_validation() for CER computation and validation logging
    - Training evaluation loops for real-time transcription display
    - Post-training evaluation scripts for text generation
    
    Character tokenizer mapping (typical):
    - ID 0: CTC blank token (filtered out)
    - ID 1: Space character
    - ID 2-27: Letters a-z  
    - ID 28: Apostrophe character
    - Unknown IDs: Filtered out for robustness
    
    Args:
        token_ids: List of character token IDs from CTC decoding
        tokenizer: CharTokenizer instance with id_to_char mapping
        
    Returns:
        Human-readable text string with characters joined together
        
    Example:
        Input: [8, 5, 12, 12, 15, 1, 23, 15, 18, 12, 4]
        Output: "hello world"  (after tokenizer mapping)
    """
    # Convert token IDs to characters using tokenizer mapping
    # Skip blank tokens (ID=0) and any unmapped token IDs
    characters: List[str] = []
    
    for token_id in token_ids:
        # Skip CTC blank tokens using named constant
        if token_id == MambaTrainingConstants.CTC_BLANK_TOKEN_ID:
            continue
            
        # Look up character for this token ID
        character = tokenizer.id_to_char.get(token_id)
        if character is not None:
            characters.append(character)
            
    return "".join(characters)


# -----------------------------
# Training / validation
# -----------------------------
@dataclass
class TrainConfig:
    """
    Training configuration dataclass with named constants and comprehensive documentation.
    
    This dataclass centralizes all training hyperparameters and configuration options
    using named constants from MambaTrainingConstants to eliminate magic numbers and
    provide clear documentation for each parameter's purpose and typical values.
    
    Used by:
    - main() function for training configuration initialization from command line arguments
    - Training loop setup for hyperparameter access throughout training process
    - Checkpoint saving operations for configuration persistence and reproducibility
    - Validation and logging systems for schedule and frequency configuration
    
    Configuration categories:
    - Dataset paths: Training and validation CSV manifest file locations
    - Training schedule: Epochs, evaluation frequency, and logging intervals
    - Model architecture: ConMamba backbone dimensions and complexity
    - Optimization: Learning rates, regularization, and gradient management
    - System optimization: Worker processes and checkpointing for Apple Silicon
    - Reproducibility: Random seed for deterministic training runs
    """
    
    # Required dataset configuration paths
    train_csv: str                      # Path to training CSV manifest (audio_path, duration, text)
    val_csv: str                        # Path to validation CSV manifest for evaluation
    
    # Training schedule configuration using named constants
    epochs: int = MambaTrainingConstants.DEFAULT_EPOCHS                       # Total training epochs for convergence
    eval_every_epochs: int = MambaTrainingConstants.DEFAULT_EVAL_EVERY_EPOCHS # Validation frequency (epochs)
    log_interval: int = MambaTrainingConstants.DEFAULT_LOG_INTERVAL           # Training step logging frequency
    
    # Model architecture configuration with Apple Silicon optimizations
    batch_size: int = MambaTrainingConstants.DEFAULT_BATCH_SIZE               # Per-device batch size for memory efficiency
    d_model: int = MambaTrainingConstants.DEFAULT_D_MODEL                     # ConMamba hidden dimension
    n_blocks: int = MambaTrainingConstants.DEFAULT_N_BLOCKS                   # ConMamba encoder block count
    
    # Optimization hyperparameters optimized for Mamba architecture  
    lr: float = MambaTrainingConstants.DEFAULT_LEARNING_RATE                  # AdamW learning rate for stable training
    weight_decay: float = MambaTrainingConstants.DEFAULT_WEIGHT_DECAY         # L2 regularization strength
    grad_clip: float = MambaTrainingConstants.DEFAULT_GRADIENT_CLIP           # Gradient clipping for stability
    
    # Apple Silicon system optimization configuration
    num_workers: int = MambaTrainingConstants.DEFAULT_NUM_WORKERS             # DataLoader workers (0 optimal for Apple Silicon)
    checkpoint_dir: str = MambaTrainingConstants.DEFAULT_CHECKPOINT_DIR       # Directory for model checkpoint storage
    
    # Training mode and reproducibility configuration
    freeze_backbone: bool = False       # Whether to freeze ConMamba backbone (projection-only training)
    seed: int = 42                      # Random seed for deterministic training reproducibility


def set_seed(seed: int) -> None:
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)


class PerformanceMonitor:
    """
    Lightweight performance monitor for analyzing data loading vs compute time balance.
    
    This monitor provides insights into training efficiency by tracking the proportion
    of time spent waiting on DataLoader vs performing training computations. It helps
    optimize num_workers and batch_size settings for Apple Silicon systems where
    I/O and compute characteristics differ from traditional CUDA systems.
    
    Used by:
    - main() training loop for real-time performance monitoring during training
    - Hyperparameter tuning workflows for optimal DataLoader worker configuration
    - Apple Silicon optimization debugging for unified memory architecture performance
    
    Monitoring phases and transitions:
    - idle: Initial state before training begins
    - data: Waiting for DataLoader batch preparation and transfer
    - train: Performing forward pass, loss computation, and backward pass
    
    Performance insights provided:
    - GPU-busy percentage: Time spent on actual model computation
    - Data-wait percentage: Time spent waiting for batch preparation
    - Balance optimization: Helps identify I/O vs compute bottlenecks
    
    Apple Silicon considerations:
    - Unified memory architecture affects data transfer patterns
    - Single-threaded I/O often optimal (num_workers=0)
    - Memory pressure from excessive workers can degrade performance
    - MPS backend synchronization affects timing accuracy
    """

    def __init__(self, log_every: int = MambaTrainingConstants.PERFORMANCE_LOG_EVERY):
        """
        Initialize performance monitor with configurable logging frequency.
        
        Args:
            log_every: Number of training steps between performance reports
                      (uses named constant optimized for Apple Silicon)
        """
        self.log_every = log_every
        self._last_time = time.perf_counter()
        self._data_wait_sum = 0.0
        self._train_sum = 0.0
        self._phase = "idle"

    def batch_fetch_started(self) -> None:
        """
        Mark the start of data loading phase for batch preparation timing.
        
        Called by:
        - Training loop immediately before DataLoader batch enumeration
        - Performance measurement code tracking I/O wait times
        
        Timing behavior:
        - Records end of previous training computation if transitioning from train phase
        - Begins measurement of DataLoader batch preparation time
        - Maintains phase state for accurate time attribution
        """
        current_time = time.perf_counter()
        if self._phase == "train":
            self._train_sum += current_time - self._last_time
        self._last_time = current_time
        self._phase = "data"

    def train_step_started(self) -> None:
        """
        Mark the start of training computation phase for model processing timing.
        
        Called by:
        - Training loop immediately after DataLoader batch retrieval
        - Performance measurement code tracking compute utilization
        
        Timing behavior:
        - Records end of data loading wait if transitioning from data phase
        - Begins measurement of forward/backward pass computation time
        - Maintains phase state for accurate time attribution
        """
        current_time = time.perf_counter()
        if self._phase == "data":
            self._data_wait_sum += current_time - self._last_time
        self._last_time = current_time
        self._phase = "train"

    def maybe_log(self, step: int) -> None:
        """
        Conditionally log performance metrics based on step interval and reset measurement window.
        
        Called by:
        - Training loop at each training step for periodic performance reporting
        - Performance monitoring systems requiring regular efficiency updates
        
        Args:
            step: Current training step number for interval calculation
            
        Reporting format:
        - GPU-busy percentage: Proportion of time spent in model computation
        - Data-wait percentage: Proportion of time spent waiting for batches
        - Step window: Number of steps covered by current measurement period
        
        Optimization insights:
        - High data-wait suggests need for more DataLoader workers (if not Apple Silicon)
        - High GPU-busy indicates efficient utilization and properly tuned I/O
        - Balanced percentages suggest optimal configuration for current hardware
        """
        if step % self.log_every != 0:
            return
            
        total_time = self._data_wait_sum + self._train_sum
        if total_time > MambaTrainingConstants.MINIMUM_TOTAL_TIME:
            # Calculate percentages using named constant for consistent scaling
            data_wait_percentage = (self._data_wait_sum / total_time) * MambaTrainingConstants.PERCENTAGE_SCALE_FACTOR
            gpu_busy_percentage = (self._train_sum / total_time) * MambaTrainingConstants.PERCENTAGE_SCALE_FACTOR
            
            print(f"  [Perf] GPU-busy: {gpu_busy_percentage:.{MambaTrainingConstants.TIME_DISPLAY_PRECISION}f}% | "
                  f"Data-wait: {data_wait_percentage:.{MambaTrainingConstants.TIME_DISPLAY_PRECISION}f}% "
                  f"(over last {self.log_every} steps)")
                  
        # Reset measurement window for next interval
        self._last_time = time.perf_counter()
        self._data_wait_sum = 0.0
        self._train_sum = 0.0
        self._phase = "idle"


def run_validation(model: nn.Module, criterion: nn.CTCLoss, loader: DataLoader, device: torch.device, tokenizer: CharTokenizer) -> Tuple[float, float]:
    model.eval()
    total_loss: float = 0.0
    total_batches: int = 0
    cer_meter = CERScore()

    with torch.no_grad():
        for feats, feat_lens, targets, target_lens, texts in loader:
            feats = feats.to(device)
            feat_lens = feat_lens.to(device)
            targets = targets.to(device)
            target_lens = target_lens.to(device)

            logits_29, out_lens = model(feats, feat_lens)          # (B, T', 29)

            # Filter invalid samples for CTC: target_len>0 and input_len>=target_len
            with torch.no_grad():
                starts = torch.cat([
                    torch.zeros(1, dtype=torch.long, device=target_lens.device),
                    target_lens[:-1].cumsum(dim=0),
                ])
                ends = starts + target_lens
                good_idx: List[int] = []
                for i in range(feats.size(0)):
                    if target_lens[i].item() > 0 and out_lens[i].item() >= target_lens[i].item():
                        good_idx.append(i)
            if len(good_idx) == 0:
                continue  # skip batch with no valid samples
            # Rebuild per-sample targets and select valid subset
            sel_targets: List[torch.Tensor] = []
            for i in good_idx:
                si = int(starts[i].item()); ei = int(ends[i].item())
                sel_targets.append(targets[si:ei])
            targets_sel = torch.cat(sel_targets) if sel_targets else targets.new_zeros(1)
            out_lens_sel = out_lens[good_idx]
            logits_sel = logits_29[good_idx]
            logp = logits_sel.log_softmax(dim=-1).transpose(0, 1)  # (T', B_sel, 29)
            tgt_lens_sel = target_lens[good_idx]

            loss = criterion(logp, targets_sel, out_lens_sel, tgt_lens_sel)
            total_loss += float(loss.item())
            total_batches += 1

            # Greedy decode and CER
            pred_ids_batch = ctc_greedy_decode(logits_29[good_idx])
            for pred_ids, ref_text in zip(pred_ids_batch, [texts[i] for i in good_idx]):
                hyp_text = ids_to_text(pred_ids, tokenizer)
                # Simple normalization: lowercase and collapse whitespace
                ref_norm = tokenizer.normalize(ref_text)
                hyp_norm = tokenizer.normalize(hyp_text)
                cer_meter.update(ref_norm, hyp_norm)

    avg_loss = total_loss / max(1, total_batches)
    return avg_loss, cer_meter.cer


def save_checkpoint(path: Path, model: nn.Module, optimizer: optim.Optimizer, cfg: TrainConfig, best_cer: float | None = None, epoch: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "state_dict": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "config": asdict(cfg),
        "best_cer": best_cer,
        "epoch": epoch,
    }
    torch.save(obj, str(path))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train Mamba-ASR with learned 1024→29 projection head (CTC)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train-csv", required=True, help="Path to training CSV manifest (path,duration,text)")
    parser.add_argument("--val-csv", required=True, help="Path to validation CSV manifest (path,duration,text)")
    parser.add_argument("--epochs", type=int, default=MambaTrainingConstants.DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=MambaTrainingConstants.DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=MambaTrainingConstants.DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=MambaTrainingConstants.DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--d-model", type=int, default=MambaTrainingConstants.DEFAULT_D_MODEL)
    parser.add_argument("--n-blocks", type=int, default=MambaTrainingConstants.DEFAULT_N_BLOCKS)
    parser.add_argument("--num-workers", type=int, default=MambaTrainingConstants.AUTO_DETECT_WORKERS, help="DataLoader workers (-1=auto-detect based on CPU cores)")
    parser.add_argument("--checkpoint-dir", type=str, default=MambaTrainingConstants.DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--eval-every-epochs", type=int, default=MambaTrainingConstants.DEFAULT_EVAL_EVERY_EPOCHS)
    parser.add_argument("--log-interval", type=int, default=MambaTrainingConstants.DEFAULT_LOG_INTERVAL)
    parser.add_argument("--grad-clip", type=float, default=MambaTrainingConstants.DEFAULT_GRADIENT_CLIP)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-post-eval", action="store_true", help="Skip projection extraction + batch eval harness at the end")

    args = parser.parse_args()

    cfg = TrainConfig(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        n_blocks=args.n_blocks,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        eval_every_epochs=args.eval_every_epochs,
        log_interval=args.log_interval,
        grad_clip=args.grad_clip,
        freeze_backbone=args.freeze_backbone,
        seed=args.seed,
    )

    set_seed(cfg.seed)
    device = get_device()
    print(f"Device: {device}")

    # Tokenizer (for CER decoding only; not passed into workers to avoid pickling issues)
    tokenizer = CharTokenizer()

    # Datasets / loaders
    train_ds = LibriSpeechCSVDataset(cfg.train_csv, sample_rate=DS.DEFAULT_SAMPLE_RATE)
    val_ds = LibriSpeechCSVDataset(cfg.val_csv, sample_rate=DS.DEFAULT_SAMPLE_RATE)

    # Auto-detect workers if requested using named constant
    worker_count = cfg.num_workers
    if worker_count == MambaTrainingConstants.AUTO_DETECT_WORKERS:
        try:
            from utils.hardware import get_optimal_worker_count
        except Exception:
            # Fallback to Apple Silicon optimized default when auto-detection fails
            worker_count = MambaTrainingConstants.FALLBACK_WORKER_COUNT
        else:
            worker_count = get_optimal_worker_count()
        print(f"Auto-detected {worker_count} dataloader workers.")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=worker_count, collate_fn=ctc_collate, pin_memory=MambaTrainingConstants.DEFAULT_PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=worker_count, collate_fn=ctc_collate, pin_memory=MambaTrainingConstants.DEFAULT_PIN_MEMORY)

    # Model
    model = MambaASRForCTC(d_model=cfg.d_model, n_blocks=cfg.n_blocks)
    if cfg.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        print("Backbone frozen; training projection head only")
    model = model.to(device)

    # Loss and optimizer using named constants
    criterion = nn.CTCLoss(blank=MambaTrainingConstants.CTC_BLANK_TOKEN_ID, zero_infinity=MambaTrainingConstants.CTC_ZERO_INFINITY)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(MambaTrainingConstants.COSINE_SCHEDULER_T_MAX_FALLBACK, cfg.epochs))

    ckpt_dir = Path(cfg.checkpoint_dir)
    last_ckpt = ckpt_dir / MambaTrainingConstants.LAST_CHECKPOINT_NAME
    best_ckpt = ckpt_dir / MambaTrainingConstants.BEST_CHECKPOINT_NAME

    best_val_cer: float | None = None

    # Training loop
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        epoch_start = time.time()
        perf = PerformanceMonitor(log_every=max(MambaTrainingConstants.MINIMUM_PERFORMANCE_LOG_INTERVAL, cfg.log_interval))

        for step, (feats, feat_lens, targets, target_lens, _) in enumerate(train_loader, start=1):
            perf.batch_fetch_started()
            feats = feats.to(device)
            feat_lens = feat_lens.to(device)
            targets = targets.to(device)
            target_lens = target_lens.to(device)

            perf.train_step_started()
            logits_29, out_lens = model(feats, feat_lens)          # (B, T', 29)

            # Filter invalid samples for CTC: target_len>0 and input_len>=target_len
            starts = torch.cat([
                torch.zeros(1, dtype=torch.long, device=target_lens.device),
                target_lens[:-1].cumsum(dim=0),
            ])
            ends = starts + target_lens
            good_idx: List[int] = []
            for i in range(feats.size(0)):
                if target_lens[i].item() > 0 and out_lens[i].item() >= target_lens[i].item():
                    good_idx.append(i)
            if len(good_idx) == 0:
                # Skip batch with no valid CTC pairs
                continue
            sel_targets: List[torch.Tensor] = []
            for i in good_idx:
                si = int(starts[i].item()); ei = int(ends[i].item())
                sel_targets.append(targets[si:ei])
            targets_sel = torch.cat(sel_targets)
            out_lens_sel = out_lens[good_idx]
            logits_sel = logits_29[good_idx]
            logp = logits_sel.log_softmax(dim=-1).transpose(0, 1)  # (T', B_sel, 29)
            tgt_lens_sel = target_lens[good_idx]

            loss = criterion(logp, targets_sel, out_lens_sel, tgt_lens_sel)

            # Guard against NaNs/Infs before zeroing previous gradients
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if math.isfinite(cfg.grad_clip) and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            epoch_losses.append(float(loss.item()))
            if step % cfg.log_interval == 0:
                avg_loss = sum(epoch_losses[-cfg.log_interval:]) / min(cfg.log_interval, len(epoch_losses))
                print(f"Epoch {epoch:02d} Step {step:05d} | Loss {avg_loss:.4f}")
            perf.maybe_log(step)

        scheduler.step()

        # End-of-epoch reporting
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.time() - epoch_start
        avg_epoch_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        print(f"Epoch {epoch:02d} done | Avg Loss {avg_epoch_loss:.4f} | Time {elapsed:.1f}s")

        # Save "last" checkpoint every epoch
        save_checkpoint(last_ckpt, model, optimizer, cfg, best_cer=best_val_cer, epoch=epoch)

        # Validation
        if (epoch % cfg.eval_every_epochs) == 0:
            val_loss, val_cer = run_validation(model, criterion, val_loader, device, tokenizer)
            print(f"Validation | Loss {val_loss:.4f} | CER {val_cer:.4f}")
            # Track best by CER
            if best_val_cer is None or val_cer < best_val_cer:
                best_val_cer = val_cer
                save_checkpoint(best_ckpt, model, optimizer, cfg, best_cer=best_val_cer, epoch=epoch)
                print(f"New best CER {best_val_cer:.4f} at epoch {epoch}; saved {best_ckpt}")

    print("Training complete.")

    if not args.no_post_eval:
        # -----------------------------
        # Post-run integration
        # 1) Extract learned projection to CSV
        # 2) Run batch evaluation harness (Core ML + Swift runner)
        # -----------------------------
        try:
            repo_root = Path(__file__).resolve().parent
            proj_out = repo_root / "exports/projection_1024x29.csv"
            ckpt_path = best_ckpt if best_ckpt.exists() else last_ckpt
            if not ckpt_path.exists():
                print("No checkpoint found for projection extraction; skipping post-run steps.")
            else:
                # Call extractor script with our parameter keys
                extractor = repo_root / "scripts/extract_projection_from_ckpt.py"
                cmd = [
                    "python3", str(extractor),
                    "--ckpt", str(ckpt_path),
                    "--w-key", "proj.weight",
                    "--b-key", "proj.bias",
                    "--out", str(proj_out),
                ]
                print("Extracting learned projection →", proj_out)
                import subprocess
                result = subprocess.run(cmd)
                if result.returncode != 0:
                    print("Projection extraction failed (non-zero exit).")
                else:
                    print("Projection CSV written:", proj_out)

                # Run batch eval harness (uses learned projection automatically)
                eval_script = repo_root / "scripts/eval_batch.sh"
                if eval_script.exists():
                    print("Running batch evaluation harness...")
                    result2 = subprocess.run(["bash", str(eval_script)])
                    if result2.returncode != 0:
                        print("Batch evaluation failed (non-zero exit). See logs above.")
                    else:
                        print("Batch evaluation complete. See exports/CoreMLTraces for reports.")
                else:
                    print("Batch eval script not found; skipping.")
        except Exception as e:
            print(f"Post-run steps encountered an error: {e}")


if __name__ == "__main__":
    main()
