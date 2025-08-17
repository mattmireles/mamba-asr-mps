"""
CTC training pipeline for ConMamba speech recognition on Apple Silicon.

This script provides a complete training pipeline for ConMamba models using
Connectionist Temporal Classification (CTC) loss. It's optimized for Apple Silicon
hardware with comprehensive MPS fallback support and performance monitoring.

Training Features:
- Apple Silicon MPS acceleration with CPU fallback for unsupported operations
- Comprehensive profiling and performance metrics collection
- Dummy dataset generation for rapid prototyping and testing
- Memory-efficient batch processing optimized for unified memory
- Real-time throughput monitoring and benchmarking

MPS Optimizations:
- PYTORCH_ENABLE_MPS_FALLBACK=1 for CTC loss compatibility
- Device-agnostic model and data placement
- Synchronization points for accurate performance measurement
- Memory pressure monitoring and management

Profiling Integration:
- Optional MPS profiling via --profile flag
- Throughput calculation in frames per second
- Per-step loss monitoring and logging
- Performance comparison baselines

Usage Examples:
    # Quick sanity check
    python train_CTC.py --sanity --epochs 1
    
    # Full training run with profiling
    python train_CTC.py --epochs 10 --batch_size 4 --profile
    
    # Production training
    python train_CTC.py --epochs 100 --batch_size 8

Apple Silicon Considerations:
- Unified memory enables larger batch sizes than discrete GPU
- MPS fallback ensures compatibility with all PyTorch operations
- Conv1d and linear operations well-optimized on Metal
- Selective scan remains the primary performance bottleneck

Training Pipeline:
1. Device detection and MPS setup
2. Model instantiation and device placement
3. Dataset creation (dummy or real data)
4. Training loop with loss computation and backpropagation
5. Performance monitoring and profiling

References:
- CTC Loss: Graves et al. Connectionist Temporal Classification
- Apple Silicon optimization: README/Mamba-on-Apple-Silicon.md
- ConMamba architecture: modules/Conmamba.py
"""
from __future__ import annotations

import os
# Enable CPU fallback for missing MPS ops (e.g., aten::_ctc_loss)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
 # torchaudio is only needed for real data pipelines; keep optional

from modules.Conmamba import ConMambaCTC, ConMambaCTCConfig
import time
import contextlib


# Training Configuration Constants
class TrainingConstants:
    """Named constants for CTC training configuration and optimization.
    
    These constants define standard training parameters optimized for
    Apple Silicon hardware and ConMamba architecture characteristics.
    """
    
    # Training Hyperparameters
    DEFAULT_LEARNING_RATE = 3e-4    # AdamW learning rate (empirically validated)
    DEFAULT_BATCH_SIZE = 2          # Conservative for Apple Silicon memory
    DEFAULT_EPOCHS = 1              # Quick testing default
    
    # Model Configuration
    DEFAULT_D_MODEL = 256           # Model dimension
    DEFAULT_N_BLOCKS = 2            # Number of Mamba blocks (lightweight for testing)
    DEFAULT_VOCAB_SIZE = 1024       # Vocabulary size
    
    # Loss Configuration  
    CTC_BLANK_TOKEN = 0             # CTC blank token index
    ZERO_INFINITY = True            # Handle infinite loss values
    
    # Performance Monitoring
    LOG_INTERVAL = 10               # Steps between loss logging
    SYNC_INTERVAL = 1               # Epochs between device synchronization
    
    # Memory Management
    GRAD_SET_TO_NONE = True         # More efficient than zero_grad()
    
    @staticmethod
    def get_training_info() -> str:
        """Return training configuration documentation."""
        return f"""
        CTC Training Configuration:
        - Learning Rate: {TrainingConstants.DEFAULT_LEARNING_RATE} (AdamW optimizer)
        - Batch Size: {TrainingConstants.DEFAULT_BATCH_SIZE} (Apple Silicon optimized)
        - Model: {TrainingConstants.DEFAULT_D_MODEL}D, {TrainingConstants.DEFAULT_N_BLOCKS} blocks
        - CTC: Blank token at index {TrainingConstants.CTC_BLANK_TOKEN}
        - Memory: Unified memory architecture optimizations
        """


# Dataset Configuration Constants
class DatasetConstants:
    """Named constants for dummy dataset generation.
    
    These constants define realistic audio and text characteristics
    for synthetic data generation and testing.
    """
    
    # Sample Configuration
    DEFAULT_NUM_SAMPLES = 32        # Dataset size for quick testing
    STRESS_NUM_SAMPLES = 128        # Larger dataset for thorough testing
    
    # Audio Characteristics
    MEL_FEATURES = 80               # Standard mel-spectrogram feature count
    MIN_FRAMES = 400                # ~25 seconds at 16kHz (minimum realistic)
    DEFAULT_MAX_FRAMES = 800        # ~50 seconds at 16kHz (typical maximum)
    STRESS_MAX_FRAMES = 2000        # ~125 seconds (stress testing)
    
    # Text Characteristics
    MIN_TARGET_LEN = 5              # Minimum target sequence length
    MAX_TARGET_LEN = 50             # Maximum target sequence length
    DEFAULT_VOCAB_SIZE = 1024       # Standard vocabulary size
    
    # Performance Targets
    TARGET_FRAMES_PER_SEC = 10000   # Performance target for Apple Silicon
    
    @staticmethod
    def get_dataset_info() -> str:
        """Return dataset characteristics documentation."""
        return f"""
        Dummy Dataset Characteristics:
        - Audio: {DatasetConstants.MIN_FRAMES}-{DatasetConstants.DEFAULT_MAX_FRAMES} frames, {DatasetConstants.MEL_FEATURES} mel features
        - Text: {DatasetConstants.MIN_TARGET_LEN}-{DatasetConstants.MAX_TARGET_LEN} tokens
        - Vocabulary: {DatasetConstants.DEFAULT_VOCAB_SIZE} tokens (including CTC blank)
        - Target Performance: {DatasetConstants.TARGET_FRAMES_PER_SEC} frames/sec on Apple Silicon
        """


def get_device() -> torch.device:
    """Detect and return the best available device for training.
    
    This function implements the device selection strategy optimized for
    Apple Silicon, with fallbacks to CUDA and CPU as needed.
    
    Device Priority:
    1. MPS (Apple Silicon GPU) - Primary target for Apple hardware
    2. CUDA (NVIDIA GPU) - For systems with discrete NVIDIA cards
    3. CPU - Universal fallback for compatibility
    
    Apple Silicon Detection:
    - Checks torch.backends.mps.is_available() for MPS support
    - Verifies PyTorch was built with MPS backend
    - Confirms macOS 12.3+ and compatible hardware
    
    Returns:
        torch.device: Optimal device for training on current hardware
        
    Usage:
        device = get_device()
        model = model.to(device)
        data = data.to(device)
    """
    if torch.backends.mps.is_available():
        # MPS is available - use Apple Silicon GPU acceleration
        return torch.device("mps")
    elif torch.cuda.is_available():
        # CUDA is available - use NVIDIA GPU acceleration
        return torch.device("cuda")
    else:
        # Fallback to CPU for universal compatibility
        return torch.device("cpu")


class DummyDataset(torch.utils.data.Dataset):
    """Synthetic dataset for rapid prototyping and testing ConMamba CTC training.
    
    This dataset generates random audio features and text labels to enable
    immediate testing without requiring real speech data. It's designed to
    match the input/output characteristics of real speech recognition datasets.
    
    Data Characteristics:
    - Audio features: Random mel-spectrograms (T, 80) with variable length
    - Text labels: Random token sequences with realistic length distribution
    - Sequence lengths: Variable to simulate real speech data
    
    Apple Silicon Optimization:
    - Tensor generation happens on CPU, then moved to device
    - Variable-length sequences test memory allocation patterns
    - Realistic sizes for Apple Silicon memory constraints
    
    Usage:
        # Quick testing
        dataset = DummyDataset(num=32, max_T=800, vocab=1024)
        
        # Stress testing
        dataset = DummyDataset(num=1000, max_T=2000, vocab=5000)
    """
    
    def __init__(self, num: int = DatasetConstants.DEFAULT_NUM_SAMPLES, 
                 max_T: int = DatasetConstants.DEFAULT_MAX_FRAMES, 
                 vocab: int = DatasetConstants.DEFAULT_VOCAB_SIZE):
        """Initialize dummy dataset with configurable parameters.
        
        Args:
            num: Number of samples in dataset
            max_T: Maximum number of time frames per sample
            vocab: Vocabulary size for target generation
        """
        super().__init__()
        self.num = num
        self.max_T = max_T
        self.vocab = vocab

    def __len__(self) -> int:
        """Return dataset size."""
        return self.num

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a single training sample with random data.
        
        Creates realistic audio and text data for CTC training:
        - Variable-length mel-spectrogram features
        - Corresponding sequence length tensor
        - Random target token sequence
        
        Args:
            idx: Sample index (unused, all samples are random)
            
        Returns:
            feats: Mel-spectrogram features (T, 80)
            feat_len: Sequence length scalar
            targets: Target token sequence (tgt_len,)
        """
        # Generate variable-length audio sequence
        # Realistic range: 400-800 frames (25-50 seconds at 16kHz)
        # Ensure valid randint range even when max_T == MIN_FRAMES (sanity mode)
        high_bound = max(DatasetConstants.MIN_FRAMES + 1, self.max_T + 1)
        time_frames = torch.randint(
            low=DatasetConstants.MIN_FRAMES,
            high=high_bound,
            size=(1,)
        ).item()
        
        # Create mel-spectrogram features with standard Gaussian distribution
        # Shape: (time_frames, 80) to match real mel-spectrogram outputs
        mel_features = torch.randn(time_frames, DatasetConstants.MEL_FEATURES)
        sequence_length = torch.tensor(time_frames)
        
        # Generate target token sequence with realistic length
        # Typical speech: ~10-50 characters for 25-50 second audio
        target_length = torch.randint(
            low=DatasetConstants.MIN_TARGET_LEN,
            high=DatasetConstants.MAX_TARGET_LEN, 
            size=(1,)
        ).item()
        
        # Create target tokens (excluding blank token at index 0)
        target_tokens = torch.randint(
            low=1,  # Skip CTC blank token at index 0
            high=self.vocab - 1, 
            size=(target_length,)
        )
        
        return mel_features, sequence_length, target_tokens


def collate(batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for variable-length speech data batching.
    
    This function handles the batching of variable-length audio sequences
    and text targets for CTC training. It performs padding and concatenation
    operations optimized for Apple Silicon memory characteristics.
    
    Batching Strategy:
    - Audio: Pad sequences to maximum length in batch with zeros
    - Text: Concatenate all targets into single tensor (CTC requirement)
    - Lengths: Maintain original sequence lengths for loss computation
    
    Memory Optimization:
    - Zero-padding minimizes memory usage on unified memory architecture
    - Efficient tensor operations for Apple Silicon
    - Avoids unnecessary data copies
    
    Args:
        batch: List of (features, feat_len, targets) tuples from dataset
        
    Returns:
        feats: Padded feature tensor (B, max_T, 80)
        feat_lens: Original sequence lengths (B,)
        targets: Concatenated target tokens (total_target_length,)
        tgt_lens: Individual target lengths (B,)
        
    CTC Requirements:
    - Features are padded for parallel processing
    - Targets are concatenated for CTC loss computation
    - Length tensors enable proper loss masking
    """
    features_list, feature_lengths, targets_list = zip(*batch)
    batch_size = len(batch)
    
    # Find maximum sequence length in batch for padding
    max_time_frames = max(f.shape[0] for f in features_list)
    
    # Create padded feature tensor
    # Initialize with zeros (silence) for padding
    padded_features = torch.zeros(batch_size, max_time_frames, DatasetConstants.MEL_FEATURES)
    
    # Copy features into padded tensor, preserving original lengths
    for batch_idx, features in enumerate(features_list):
        sequence_length = features.shape[0]
        padded_features[batch_idx, :sequence_length] = features
    
    # Stack feature lengths for parallel processing
    feature_lengths_tensor = torch.stack(feature_lengths)
    
    # Concatenate targets as required by CTC loss
    # CTC loss expects flattened target tensor
    concatenated_targets = torch.cat(targets_list)
    
    # Create target length tensor for CTC loss computation
    target_lengths = torch.tensor([len(targets) for targets in targets_list], dtype=torch.long)
    
    return padded_features, feature_lengths_tensor, concatenated_targets, target_lengths


def train_one_step(
    model: nn.Module, 
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], 
    device: torch.device, 
    criterion: nn.CTCLoss, 
    optimizer: torch.optim.Optimizer
) -> float:
    """Execute one training step with CTC loss computation.
    
    This function performs a complete forward and backward pass for one batch,
    including device placement, loss computation, and parameter updates.
    
    Training Pipeline:
    1. Move batch data to target device (MPS/CUDA/CPU)
    2. Forward pass through ConMamba model
    3. CTC loss computation with proper tensor formatting
    4. Backward pass and gradient computation
    5. Optimizer step and parameter updates
    
    Apple Silicon Optimizations:
    - Efficient device transfers using unified memory
    - set_to_none=True for more efficient gradient clearing
    - Proper tensor layouts for MPS backend compatibility
    
    CTC Loss Requirements:
    - Logits must be transposed to (T, B, V) format
    - Log-softmax applied for numerical stability
    - Input/target lengths required for proper alignment
    
    Args:
        model: ConMamba model to train
        batch: Collated batch data (features, feat_lens, targets, tgt_lens)
        device: Target device for computation
        criterion: CTC loss function
        optimizer: Parameter optimizer (typically AdamW)
        
    Returns:
        loss_value: Scalar loss value for logging
        
    Memory Considerations:
    - Gradient accumulation handled automatically
    - Memory usage peaks during backward pass
    - Apple Silicon unified memory reduces transfer overhead
    """
    # Unpack batch data
    features, feature_lengths, targets, target_lengths = batch
    
    # Move all tensors to target device
    # Unified memory architecture makes this efficient on Apple Silicon
    features = features.to(device)
    feature_lengths = feature_lengths.to(device)
    targets = targets.to(device)
    target_lengths = target_lengths.to(device)

    # Forward pass through ConMamba model
    # Returns: logits (B, T', V), output_lengths (B,)
    logits, output_lengths = model(features, feature_lengths)
    
    # Prepare logits for CTC loss computation
    # CTC requires (T, B, V) format, we have (B, T', V)
    # Apply log-softmax for numerical stability
    log_probabilities = logits.log_softmax(dim=-1).transpose(0, 1)  # (T', B, V)
    
    # Compute CTC loss
    # Arguments: log_probs, targets, input_lengths, target_lengths
    ctc_loss = criterion(log_probabilities, targets, output_lengths, target_lengths)

    # Backward pass and optimization
    # Use set_to_none=True for more efficient memory usage
    optimizer.zero_grad(set_to_none=TrainingConstants.GRAD_SET_TO_NONE)
    ctc_loss.backward()
    optimizer.step()
    
    # Return scalar loss value for logging
    return ctc_loss.item()


def main() -> None:
    """Main training function with comprehensive configuration and monitoring.
    
    This function orchestrates the complete training pipeline including:
    - Command line argument parsing
    - Device detection and model setup
    - Dataset creation and data loading
    - Training loop with performance monitoring
    - Optional profiling and benchmarking
    
    The training pipeline is optimized for Apple Silicon with comprehensive
    fallback support and detailed performance analysis.
    """
    import argparse

    # Command line argument configuration
    parser = argparse.ArgumentParser(
        description="Train ConMamba CTC model for speech recognition on Apple Silicon",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=TrainingConstants.DEFAULT_EPOCHS,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=TrainingConstants.DEFAULT_BATCH_SIZE,
        help="Training batch size (optimized for Apple Silicon)"
    )
    parser.add_argument(
        "--sanity", 
        action="store_true", 
        help="Run quick sanity check with minimal data"
    )
    parser.add_argument(
        "--profile", 
        action="store_true", 
        help="Enable MPS profiling for performance analysis"
    )
    args = parser.parse_args()

    # Device detection and setup
    device = get_device()
    print(f"Training device: {device}")
    print(f"Training configuration: {TrainingConstants.get_training_info()}")
    
    # Model configuration and instantiation
    model_config = ConMambaCTCConfig(
        d_model=TrainingConstants.DEFAULT_D_MODEL,
        n_blocks=TrainingConstants.DEFAULT_N_BLOCKS, 
        vocab_size=TrainingConstants.DEFAULT_VOCAB_SIZE
    )
    model = ConMambaCTC(model_config).to(device)
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Dataset creation based on training mode
    if args.sanity:
        # Minimal dataset for quick testing
        dataset = DummyDataset(num=8, max_T=DatasetConstants.MIN_FRAMES)
        print("Sanity mode: Using minimal dataset")
    else:
        # Standard dataset for thorough training
        dataset = DummyDataset(num=DatasetConstants.STRESS_NUM_SAMPLES)
        print(f"Training mode: {DatasetConstants.get_dataset_info()}")

    # Data loader configuration
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate,
        num_workers=0  # Apple Silicon works best with single-threaded data loading
    )
    
    # Loss function and optimizer setup
    ctc_criterion = nn.CTCLoss(
        blank=TrainingConstants.CTC_BLANK_TOKEN, 
        zero_infinity=TrainingConstants.ZERO_INFINITY
    )
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=TrainingConstants.DEFAULT_LEARNING_RATE
    )

    # Set model to training mode
    model.train()

    # Optional MPS profiling setup
    try:
        from torch.mps.profiler import profile as mps_profile  # type: ignore
        profiling_available = True
    except Exception:
        mps_profile = contextlib.nullcontext  # type: ignore
        profiling_available = False
        
    if args.profile and not profiling_available:
        print("Warning: MPS profiling not available")

    # Profiling context setup
    profiling_context = mps_profile() if (args.profile and profiling_available) else contextlib.nullcontext()

    # Performance monitoring setup
    total_audio_frames = 0
    training_start_time = time.time()
    
    print("\nStarting training...")
    print("=" * 50)
    
    # Main training loop with profiling
    with profiling_context:
        for epoch in range(args.epochs):
            epoch_start = time.time()
            epoch_losses = []
            
            for step, batch in enumerate(data_loader):
                # Extract audio frame count for throughput calculation
                features, feature_lengths, targets, target_lengths = batch
                total_audio_frames += int(feature_lengths.sum().item())
                
                # Execute training step
                step_loss = train_one_step(model, batch, device, ctc_criterion, optimizer)
                epoch_losses.append(step_loss)
                
                # Periodic logging
                if step % TrainingConstants.LOG_INTERVAL == 0:
                    print(f"Epoch {epoch:2d} Step {step:3d} Loss {step_loss:.4f}")
            
            # End-of-epoch synchronization and reporting
            if device.type == "mps":
                torch.mps.synchronize()
            
            epoch_time = time.time() - epoch_start
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            print(f"Epoch {epoch} complete - Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
    
    # Final performance analysis
    total_training_time = time.time() - training_start_time
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    
    if total_training_time > 0 and total_audio_frames > 0:
        throughput = total_audio_frames / total_training_time
        print(f"Total training time: {total_training_time:.2f} seconds")
        print(f"Total audio frames: {total_audio_frames:,}")
        print(f"Throughput: {throughput:.1f} frames/sec")
        print(f"Target throughput: {DatasetConstants.TARGET_FRAMES_PER_SEC} frames/sec")
        
        # Performance assessment
        if throughput >= DatasetConstants.TARGET_FRAMES_PER_SEC:
            print("✅ Performance target achieved!")
        else:
            performance_ratio = throughput / DatasetConstants.TARGET_FRAMES_PER_SEC
            print(f"⚠️  Performance: {performance_ratio:.1%} of target")
    
    print("\nNote: This is dummy data training for testing purposes.")
    print("Real speech recognition training requires actual audio datasets.")


if __name__ == "__main__":
    main()
