#!/usr/bin/env python3
"""Train the 29-logit ConMamba CTC model on real speech.

The character head is part of :class:`modules.Conmamba.ConMambaCTC`; there is
no intermediate 1024-class vocabulary or post-hoc projection. Checkpoints under
``checkpoints/`` contain the exact model consumed by
``scripts/export_coreml.py``.
"""
from __future__ import annotations

import os
import math
import time
import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

# Enable CPU fallback for missing MPS ops (e.g., aten::_ctc_loss)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler

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
    
    # Vocabulary size matches CharTokenizer exactly.
    CHARACTER_VOCAB_SIZE = 29           # blank + space + a-z + apostrophe
    CTC_BLANK_TOKEN_ID = 0              # CTC blank token identifier (standard CTC convention)
    
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
    DEFAULT_CHECKPOINT_DIR = "checkpoints"  # Gitignored local checkpoints
    
    # MARK: - Data Processing Constants
    
    # CTC processing constants for loss computation and decoding
    CTC_ZERO_INFINITY = False           # Expose impossible alignments; filter them explicitly
    COSINE_SCHEDULER_T_MAX_FALLBACK = 1 # Minimum T_max for cosine annealing scheduler
    
    # Validation and error handling constants
    MINIMUM_LOSS_BATCHES = 1            # Minimum batches required for loss averaging
    MINIMUM_TOTAL_TIME = 0.0            # Minimum time threshold for performance calculations
    
    # Text processing constants for character-level tokenization
    BLANK_CHAR_TOKEN_ID = 0             # Character tokenizer blank token identifier
    SPACE_TOKEN_ID = 1                  # Character tokenizer space token identifier (typical mapping)
    
# -----------------------------
# Utility: device selection (MPS → CUDA → CPU)
# -----------------------------
def get_device(preference: str = "auto") -> torch.device:
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
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
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


class DurationBucketBatchSampler(Sampler[List[int]]):
    """Shuffle globally, then batch similar-duration utterances.

    Sorting only within shuffled pools retains stochastic batches without
    paying the full padding cost of unrelated short and long clips.
    """

    def __init__(
        self,
        durations: List[float],
        batch_size: int,
        bucket_size_multiplier: int,
        seed: int,
    ):
        self.durations = durations
        self.batch_size = batch_size
        self.pool_size = batch_size * bucket_size_multiplier
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[List[int]]:
        generator = random.Random(self.seed + self.epoch)
        self.epoch += 1
        indices = list(range(len(self.durations)))
        generator.shuffle(indices)
        batches: List[List[int]] = []
        for start in range(0, len(indices), self.pool_size):
            pool = indices[start : start + self.pool_size]
            pool.sort(key=self.durations.__getitem__)
            batches.extend(
                pool[offset : offset + self.batch_size]
                for offset in range(0, len(pool), self.batch_size)
            )
        generator.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return math.ceil(len(self.durations) / self.batch_size)


def ctc_valid_indices(
    targets: torch.Tensor,
    target_lens: torch.Tensor,
    output_lens: torch.Tensor,
) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
    """Return samples whose CTC paths are mathematically possible.

    CTC needs one extra input timestep for every adjacent repeated target
    symbol, not merely ``output_len >= target_len``. This function is shared
    by training and validation so ``zero_infinity`` does not silently hide
    an impossible alignment.
    """
    starts = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=target_lens.device),
            target_lens[:-1].cumsum(dim=0),
        ]
    )
    ends = starts + target_lens
    valid: List[int] = []
    for index in range(target_lens.numel()):
        target_length = int(target_lens[index].item())
        if target_length <= 0:
            continue
        start = int(starts[index].item())
        end = int(ends[index].item())
        target = targets[start:end]
        adjacent_repeats = (
            int((target[1:] == target[:-1]).sum().item())
            if target_length > 1
            else 0
        )
        minimum_input_length = target_length + adjacent_repeats
        if int(output_lens[index].item()) >= minimum_input_length:
            valid.append(index)
    return valid, starts, ends


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


@dataclass
class WERScore:
    total_words: int = 0
    total_errors: int = 0

    def update(self, ref: str, hyp: str) -> None:
        reference = ref.split()
        hypothesis = hyp.split()
        rows, columns = len(reference), len(hypothesis)
        distance = [[0] * (columns + 1) for _ in range(rows + 1)]
        for row in range(rows + 1):
            distance[row][0] = row
        for column in range(columns + 1):
            distance[0][column] = column
        for row in range(1, rows + 1):
            for column in range(1, columns + 1):
                cost = 0 if reference[row - 1] == hypothesis[column - 1] else 1
                distance[row][column] = min(
                    distance[row - 1][column] + 1,
                    distance[row][column - 1] + 1,
                    distance[row - 1][column - 1] + cost,
                )
        self.total_errors += distance[rows][columns]
        self.total_words += max(rows, 1)

    @property
    def wer(self) -> float:
        return self.total_errors / max(1, self.total_words)


def ctc_greedy_decode(
    logits_29: torch.Tensor,
    output_lens: torch.Tensor | None = None,
    blank_id: int = MambaTrainingConstants.CTC_BLANK_TOKEN_ID,
) -> List[List[int]]:
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
                   Output directly by ConMambaCTC
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
    for index, token_sequence in enumerate(predicted_tokens):
        if output_lens is not None:
            token_sequence = token_sequence[: int(output_lens[index].item())]
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
    max_duration: float = 0.0           # 0 keeps every manifest row
    
    # Training schedule configuration using named constants
    epochs: int = MambaTrainingConstants.DEFAULT_EPOCHS                       # Total training epochs for convergence
    eval_every_epochs: int = MambaTrainingConstants.DEFAULT_EVAL_EVERY_EPOCHS # Validation frequency (epochs)
    log_interval: int = MambaTrainingConstants.DEFAULT_LOG_INTERVAL           # Training step logging frequency
    max_steps: int = 0                    # Global debug cap; 0 runs full epochs
    target_wer: float = 0.25              # Stop after validation reaches this gate
    
    # Model architecture configuration with Apple Silicon optimizations
    batch_size: int = MambaTrainingConstants.DEFAULT_BATCH_SIZE               # Per-device batch size for memory efficiency
    bucket_size_multiplier: int = 50      # Similar-duration pool size in batches
    d_model: int = MambaTrainingConstants.DEFAULT_D_MODEL                     # ConMamba hidden dimension
    n_blocks: int = MambaTrainingConstants.DEFAULT_N_BLOCKS                   # ConMamba encoder block count
    
    # Optimization hyperparameters optimized for Mamba architecture  
    lr: float = MambaTrainingConstants.DEFAULT_LEARNING_RATE                  # AdamW learning rate for stable training
    weight_decay: float = MambaTrainingConstants.DEFAULT_WEIGHT_DECAY         # L2 regularization strength
    grad_clip: float = MambaTrainingConstants.DEFAULT_GRADIENT_CLIP           # Gradient clipping for stability
    scheduler: str = "cosine"           # cosine or constant
    
    # Apple Silicon system optimization configuration
    num_workers: int = MambaTrainingConstants.DEFAULT_NUM_WORKERS             # DataLoader workers (0 optimal for Apple Silicon)
    checkpoint_dir: str = MambaTrainingConstants.DEFAULT_CHECKPOINT_DIR       # Directory for model checkpoint storage
    metrics_csv: str = ""                # Empty derives checkpoints/metrics.csv
    
    # Reproducibility configuration
    seed: int = 42                      # Random seed for deterministic training reproducibility
    device: str = "auto"                # auto, mps, or cpu


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


def run_validation(
    model: nn.Module,
    criterion: nn.CTCLoss,
    loader: DataLoader,
    device: torch.device,
    tokenizer: CharTokenizer,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss: float = 0.0
    total_batches: int = 0
    cer_meter = CERScore()
    wer_meter = WERScore()

    with torch.no_grad():
        for feats, feat_lens, targets, target_lens, texts in loader:
            feats = feats.to(device)
            feat_lens = feat_lens.to(device)
            targets = targets.to(device)
            target_lens = target_lens.to(device)

            logits_29, out_lens = model(feats, feat_lens)          # (B, T', 29)

            with torch.no_grad():
                good_idx, starts, ends = ctc_valid_indices(
                    targets,
                    target_lens,
                    out_lens,
                )
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
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("validation produced a non-finite CTC loss")
            total_loss += float(loss.item())
            total_batches += 1

            # Greedy decode and CER
            pred_ids_batch = ctc_greedy_decode(
                logits_29[good_idx],
                out_lens[good_idx],
            )
            for pred_ids, ref_text in zip(pred_ids_batch, [texts[i] for i in good_idx]):
                hyp_text = ids_to_text(pred_ids, tokenizer)
                # Simple normalization: lowercase and collapse whitespace
                ref_norm = tokenizer.normalize(ref_text)
                hyp_norm = tokenizer.normalize(hyp_text)
                cer_meter.update(ref_norm, hyp_norm)
                wer_meter.update(ref_norm, hyp_norm)

    avg_loss = total_loss / max(1, total_batches)
    return avg_loss, cer_meter.cer, wer_meter.wer


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    cfg: TrainConfig,
    best_cer: float | None = None,
    best_wer: float | None = None,
    epoch: int | None = None,
    global_step: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "state_dict": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": asdict(cfg),
        "best_cer": best_cer,
        "best_wer": best_wer,
        "epoch": epoch,
        "global_step": global_step,
    }
    torch.save(obj, str(path))


def append_metrics_row(path: Path, row: dict) -> None:
    """Append one epoch record to a local, resumable metrics CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the direct 29-logit ConMamba CTC model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-csv", required=True, help="Path to training CSV manifest (path,duration,text)")
    parser.add_argument("--val-csv", required=True, help="Path to validation CSV manifest (path,duration,text)")
    parser.add_argument("--max-duration", type=float, default=0.0, help="Optional duration cap in seconds (0 keeps the full manifest)")
    parser.add_argument("--epochs", type=int, default=MambaTrainingConstants.DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=MambaTrainingConstants.DEFAULT_BATCH_SIZE)
    parser.add_argument("--bucket-size-multiplier", type=int, default=50, help="Duration-sorted pool size in batches (0 disables bucketing)")
    parser.add_argument("--lr", type=float, default=MambaTrainingConstants.DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=MambaTrainingConstants.DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--scheduler", choices=["cosine", "constant"], default="cosine")
    parser.add_argument("--d-model", type=int, default=MambaTrainingConstants.DEFAULT_D_MODEL)
    parser.add_argument("--n-blocks", type=int, default=MambaTrainingConstants.DEFAULT_N_BLOCKS)
    parser.add_argument("--num-workers", type=int, default=MambaTrainingConstants.AUTO_DETECT_WORKERS, help="DataLoader workers (-1=auto-detect based on CPU cores)")
    parser.add_argument("--checkpoint-dir", type=str, default=MambaTrainingConstants.DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--metrics-csv", type=str, default="", help="Epoch metrics CSV (default: <checkpoint-dir>/metrics.csv)")
    parser.add_argument("--resume", type=str, default="", help="Resume model/optimizer/scheduler from a checkpoint")
    parser.add_argument("--reset-optimizer", action="store_true", help="When resuming, retain model/best metrics but start a fresh optimizer and scheduler")
    parser.add_argument("--eval-every-epochs", type=int, default=MambaTrainingConstants.DEFAULT_EVAL_EVERY_EPOCHS)
    parser.add_argument("--log-interval", type=int, default=MambaTrainingConstants.DEFAULT_LOG_INTERVAL)
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after this many global batches (0=unlimited)")
    parser.add_argument("--target-wer", type=float, default=0.25, help="Stop after validation reaches this WER (negative disables)")
    parser.add_argument("--grad-clip", type=float, default=MambaTrainingConstants.DEFAULT_GRADIENT_CLIP)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")

    args = parser.parse_args()

    cfg = TrainConfig(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        max_duration=max(0.0, args.max_duration),
        epochs=args.epochs,
        batch_size=args.batch_size,
        bucket_size_multiplier=max(0, args.bucket_size_multiplier),
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        d_model=args.d_model,
        n_blocks=args.n_blocks,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        metrics_csv=args.metrics_csv,
        eval_every_epochs=args.eval_every_epochs,
        log_interval=args.log_interval,
        max_steps=max(0, args.max_steps),
        target_wer=args.target_wer,
        grad_clip=args.grad_clip,
        seed=args.seed,
        device=args.device,
    )

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"Device: {device}")

    # Tokenizer (for CER decoding only; not passed into workers to avoid pickling issues)
    tokenizer = CharTokenizer()

    # Datasets / loaders
    train_ds = LibriSpeechCSVDataset(
        cfg.train_csv,
        sample_rate=DS.DEFAULT_SAMPLE_RATE,
        max_duration=cfg.max_duration,
    )
    val_ds = LibriSpeechCSVDataset(
        cfg.val_csv,
        sample_rate=DS.DEFAULT_SAMPLE_RATE,
        max_duration=cfg.max_duration,
    )

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

    train_batch_sampler: DurationBucketBatchSampler | None = None
    if cfg.bucket_size_multiplier > 0:
        train_batch_sampler = DurationBucketBatchSampler(
            durations=[duration for _, duration, _ in train_ds.rows],
            batch_size=cfg.batch_size,
            bucket_size_multiplier=cfg.bucket_size_multiplier,
            seed=cfg.seed,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            num_workers=worker_count,
            collate_fn=ctc_collate,
            pin_memory=MambaTrainingConstants.DEFAULT_PIN_MEMORY,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=worker_count,
            collate_fn=ctc_collate,
            pin_memory=MambaTrainingConstants.DEFAULT_PIN_MEMORY,
        )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=worker_count, collate_fn=ctc_collate, pin_memory=MambaTrainingConstants.DEFAULT_PIN_MEMORY)

    # Model
    model = ConMambaCTC(
        ConMambaCTCConfig(
            d_model=cfg.d_model,
            n_blocks=cfg.n_blocks,
            vocab_size=MambaTrainingConstants.CHARACTER_VOCAB_SIZE,
        )
    )
    model = model.to(device)

    # Loss and optimizer using named constants
    criterion = nn.CTCLoss(blank=MambaTrainingConstants.CTC_BLANK_TOKEN_ID, zero_infinity=MambaTrainingConstants.CTC_ZERO_INFINITY)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.scheduler == "constant":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(
                MambaTrainingConstants.COSINE_SCHEDULER_T_MAX_FALLBACK,
                cfg.epochs,
            ),
        )

    ckpt_dir = Path(cfg.checkpoint_dir)
    last_ckpt = ckpt_dir / MambaTrainingConstants.LAST_CHECKPOINT_NAME
    best_ckpt = ckpt_dir / MambaTrainingConstants.BEST_CHECKPOINT_NAME
    metrics_path = Path(cfg.metrics_csv) if cfg.metrics_csv else ckpt_dir / "metrics.csv"

    best_val_cer: float | None = None
    best_val_wer: float | None = None
    start_epoch = 1
    global_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise SystemExit(f"resume checkpoint not found: {resume_path}")
        try:
            resume = torch.load(resume_path, map_location=device, weights_only=True)
        except TypeError:
            resume = torch.load(resume_path, map_location=device)
        model.load_state_dict(resume["state_dict"], strict=True)
        best_val_cer = resume.get("best_cer")
        best_val_wer = resume.get("best_wer")
        start_epoch = int(resume.get("epoch") or 0) + 1
        global_step = int(resume.get("global_step") or 0)
        if args.reset_optimizer:
            optimizer = optim.AdamW(
                filter(lambda parameter: parameter.requires_grad, model.parameters()),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
            if cfg.scheduler == "constant":
                scheduler = optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lambda _: 1.0,
                )
            else:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(
                        MambaTrainingConstants.COSINE_SCHEDULER_T_MAX_FALLBACK,
                        cfg.epochs - start_epoch + 1,
                    ),
                )
            print(
                f"Reset optimizer/scheduler at lr={cfg.lr:g} "
                f"scheduler={cfg.scheduler}"
            )
        else:
            optimizer.load_state_dict(resume["optim_state"])
            if "scheduler_state" in resume:
                scheduler.load_state_dict(resume["scheduler_state"])
        print(
            f"Resumed {resume_path} at epoch={start_epoch} "
            f"global_step={global_step}"
        )
    if train_batch_sampler is not None:
        train_batch_sampler.epoch = start_epoch - 1

    # Training loop
    nonfinite_loss_count = 0
    nonfinite_gradient_count = 0
    invalid_sample_count = 0
    empty_valid_batch_count = 0
    stop_training = False
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        epoch_start = time.time()
        perf = PerformanceMonitor(log_every=max(MambaTrainingConstants.MINIMUM_PERFORMANCE_LOG_INTERVAL, cfg.log_interval))

        for step, (feats, feat_lens, targets, target_lens, _) in enumerate(train_loader, start=1):
            global_step += 1
            if cfg.max_steps and global_step > cfg.max_steps:
                global_step -= 1
                stop_training = True
                break
            perf.batch_fetch_started()
            feats = feats.to(device)
            feat_lens = feat_lens.to(device)
            targets = targets.to(device)
            target_lens = target_lens.to(device)

            perf.train_step_started()
            logits_29, out_lens = model(feats, feat_lens)          # (B, T', 29)

            good_idx, starts, ends = ctc_valid_indices(
                targets,
                target_lens,
                out_lens,
            )
            invalid_sample_count += feats.size(0) - len(good_idx)
            if len(good_idx) == 0:
                # Skip batch with no valid CTC pairs
                empty_valid_batch_count += 1
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
            if not bool(torch.isfinite(loss).item()):
                nonfinite_loss_count += 1
                raise RuntimeError(
                    f"non-finite CTC loss at global_step={global_step} "
                    f"epoch={epoch} batch_step={step}; refusing to skip"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_limit = (
                cfg.grad_clip
                if math.isfinite(cfg.grad_clip) and cfg.grad_clip > 0
                else float("inf")
            )
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_limit,
            )
            if not bool(torch.isfinite(gradient_norm).item()):
                nonfinite_gradient_count += 1
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError(
                    f"non-finite gradient norm at global_step={global_step} "
                    f"epoch={epoch} batch_step={step}; refusing to step"
                )
            optimizer.step()

            epoch_losses.append(float(loss.item()))
            if step % cfg.log_interval == 0:
                recent_losses = epoch_losses[-cfg.log_interval:]
                if recent_losses:
                    avg_loss = sum(recent_losses) / len(recent_losses)
                    print(
                        f"Epoch {epoch:02d} Step {step:05d} | "
                        f"Loss {avg_loss:.4f} | "
                        f"GradNormPreClip {float(gradient_norm.item()):.4f}"
                    )
            perf.maybe_log(step)
            if cfg.max_steps and global_step >= cfg.max_steps:
                stop_training = True
                break

        scheduler.step()

        # End-of-epoch reporting
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.time() - epoch_start
        avg_epoch_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        print(f"Epoch {epoch:02d} done | Avg Loss {avg_epoch_loss:.4f} | Time {elapsed:.1f}s")

        # Validation
        val_loss: float | None = None
        val_cer: float | None = None
        val_wer: float | None = None
        if (epoch % cfg.eval_every_epochs) == 0:
            val_loss, val_cer, val_wer = run_validation(
                model,
                criterion,
                val_loader,
                device,
                tokenizer,
            )
            print(
                f"Validation | Loss {val_loss:.4f} | "
                f"CER {val_cer:.4f} | WER {val_wer:.4f}"
            )
            improved = (
                best_val_wer is None
                or val_wer < best_val_wer
                or (
                    val_wer == best_val_wer
                    and (best_val_cer is None or val_cer < best_val_cer)
                )
            )
            if improved:
                best_val_cer = val_cer
                best_val_wer = val_wer
                save_checkpoint(
                    best_ckpt,
                    model,
                    optimizer,
                    scheduler,
                    cfg,
                    best_cer=best_val_cer,
                    best_wer=best_val_wer,
                    epoch=epoch,
                    global_step=global_step,
                )
                print(
                    f"New best WER {best_val_wer:.4f} "
                    f"(CER {best_val_cer:.4f}) at epoch {epoch}; "
                    f"saved {best_ckpt}"
                )
            if cfg.target_wer >= 0 and val_wer <= cfg.target_wer:
                stop_training = True
                print(
                    f"Target WER reached: {val_wer:.4f} "
                    f"<= {cfg.target_wer:.4f}"
                )

        append_metrics_row(
            metrics_path,
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": f"{avg_epoch_loss:.8f}",
                "val_loss": "" if val_loss is None else f"{val_loss:.8f}",
                "val_cer": "" if val_cer is None else f"{val_cer:.8f}",
                "val_wer": "" if val_wer is None else f"{val_wer:.8f}",
                "learning_rate": f"{optimizer.param_groups[0]['lr']:.10g}",
                "epoch_seconds": f"{elapsed:.3f}",
                "nonfinite_losses": nonfinite_loss_count,
                "nonfinite_gradients": nonfinite_gradient_count,
                "invalid_samples": invalid_sample_count,
            },
        )
        save_checkpoint(
            last_ckpt,
            model,
            optimizer,
            scheduler,
            cfg,
            best_cer=best_val_cer,
            best_wer=best_val_wer,
            epoch=epoch,
            global_step=global_step,
        )

        if stop_training:
            break

    print(
        "Training complete. "
        f"global_steps={global_step} "
        f"nonfinite_losses={nonfinite_loss_count} "
        f"nonfinite_gradients={nonfinite_gradient_count} "
        f"invalid_samples={invalid_sample_count} "
        f"empty_valid_batches={empty_valid_batch_count}"
    )


if __name__ == "__main__":
    main()
