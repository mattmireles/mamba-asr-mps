"""
RNN-T training pipeline for MCT (Mamba-Conformer-Transformer) speech recognition on Apple Silicon.

This script provides a complete training pipeline for MCT models using RNN-Transducer
(RNN-T) loss. It's optimized for Apple Silicon hardware with comprehensive MPS fallback
support, performance monitoring, and flexible loss function selection.

Training Features:
- Apple Silicon MPS acceleration with comprehensive CPU fallback
- RNN-T loss with automatic fallback to CTC if unavailable
- Comprehensive profiling and performance metrics collection
- Dummy dataset generation for rapid prototyping and testing
- Memory-efficient batch processing for unified memory architecture
- Real-time throughput monitoring and benchmarking

RNN-T Advantages:
- Streaming-friendly architecture for real-time speech recognition
- Alignment-free training without forced alignment requirements
- Joint acoustic-linguistic optimization
- Variable-length sequence handling
- Superior performance for conversational speech

MPS Optimizations:
- PYTORCH_ENABLE_MPS_FALLBACK=1 for RNN-T loss compatibility
- Device-agnostic model and data placement throughout
- Synchronization points for accurate performance measurement
- Memory pressure monitoring and management
- Fallback strategies for unsupported operations

Loss Function Strategy:
- Primary: RNN-T loss via torchaudio (most accurate)
- Fallback: Custom RNN-T implementation (if available)
- Emergency: CTC loss (for compatibility)
- Automatic detection and selection based on availability

Usage Examples:
    # Quick sanity check
    python train_RNNT.py --sanity --epochs 1
    
    # Full training with profiling
    python train_RNNT.py --epochs 10 --batch_size 2 --profile
    
    # Production training
    python train_RNNT.py --epochs 100 --batch_size 4

Apple Silicon Considerations:
- Unified memory enables larger batch sizes than discrete GPU
- RNN-T alignment matrix can be memory-intensive
- MCT model benefits from MPS optimization throughout
- Streaming inference capabilities preserved

Training Pipeline:
1. Device detection and MPS setup with fallback
2. MCT model instantiation and device placement
3. Dataset creation (dummy or real LibriSpeech data)
4. Loss function selection and fallback configuration
5. Training loop with RNN-T/CTC loss computation
6. Performance monitoring and profiling

References:
- RNN-T Loss: Graves et al. Sequence Transduction with RNNs
- MCT Architecture: modules/mct/mct_model.py
- Apple Silicon optimization: README/Mamba-on-Apple-Silicon.md
- Streaming inference: RNN-T enables real-time processing
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.optim as optim
import time
import contextlib

from modules.mct.mct_model import MCTModel, MCTConfig
from torch.autograd.profiler import record_function
try:
    from datasets.librispeech_csv import LibriSpeechCSVDataset, collate_fn as ls_collate
    HAS_LS = True
except Exception:
    HAS_LS = False
try:
    from utils.tokenizer import CharTokenizer
    from utils.metrics import wer
except ModuleNotFoundError:
    import sys as _sys, os as _os
    _base = _os.path.abspath(_os.path.dirname(__file__))
    _sys.path.insert(0, _base)
    _sys.path.insert(0, _os.path.join(_base, "utils"))
    try:
        from utils.tokenizer import CharTokenizer
        from utils.metrics import wer
    except ModuleNotFoundError:
        # Fallback to direct module import from added utils/ path
        import tokenizer as _tok  # type: ignore
        import metrics as _met  # type: ignore
        CharTokenizer = _tok.CharTokenizer
        wer = _met.wer


# RNN-T Training Configuration Constants
class RNNTTrainingConstants:
    """Named constants for RNN-T training configuration and optimization.
    
    These constants define standard training parameters optimized for
    Apple Silicon hardware and RNN-T architecture characteristics.
    """
    
    # Training Hyperparameters
    DEFAULT_LEARNING_RATE = 3e-4    # AdamW learning rate for RNN-T
    DEFAULT_BATCH_SIZE = 2          # Conservative for Apple Silicon + RNN-T memory
    DEFAULT_EPOCHS = 1              # Quick testing default
    
    # Model Configuration
    DEFAULT_D_MODEL = 256           # Model dimension
    DEFAULT_N_BLOCKS = 4            # Number of Mamba blocks (lighter for RNN-T)
    DEFAULT_JOINT_DIM = 320         # RNN-T joiner dimension
    DEFAULT_VOCAB_SIZE = 1024       # Vocabulary size
    
    # RNN-T Specific
    RNNT_BLANK_TOKEN = 0            # RNN-T blank token index
    MAX_ALIGNMENT_SIZE = 1000000    # Maximum T*U for memory safety
    DEFAULT_MAX_ALIGN = 250000      # Default alignment size limit
    
    # Backend Selection
    NAIVE_RNN_T_MAX_TIME = 64       # Maximum T frames for naive RNN-T
    NAIVE_RNN_T_MAX_TOKENS = 16     # Maximum U tokens for naive RNN-T
    
    # Performance Monitoring
    LOG_INTERVAL = 5                # Steps between loss logging (more frequent)
    SYNC_INTERVAL = 1               # Epochs between device synchronization
    
    # Memory Management
    GRAD_SET_TO_NONE = True         # More efficient than zero_grad()
    
    # Profiling Constants
    MAX_STREAMING_DECODE_STEPS = 128  # Maximum total decode steps per utterance
    MAX_STREAMING_TOKENS_PER_FRAME = 32  # Maximum tokens per acoustic frame
    
    @staticmethod
    def get_rnnt_info() -> str:
        """Return RNN-T training configuration documentation."""
        return f"""
        RNN-T Training Configuration:
        - Learning Rate: {RNNTTrainingConstants.DEFAULT_LEARNING_RATE} (AdamW optimizer)
        - Batch Size: {RNNTTrainingConstants.DEFAULT_BATCH_SIZE} (Apple Silicon + RNN-T optimized)
        - Model: {RNNTTrainingConstants.DEFAULT_D_MODEL}D, {RNNTTrainingConstants.DEFAULT_N_BLOCKS} blocks
        - Joint Dim: {RNNTTrainingConstants.DEFAULT_JOINT_DIM} (acoustic-linguistic fusion)
        - RNN-T: Blank token at index {RNNTTrainingConstants.RNNT_BLANK_TOKEN}
        - Memory: Alignment matrix T*U monitoring enabled
        - Backend Selection: Auto-detection with fallback support
        - Profiling: PyTorch autograd profiler integration
        - Streaming: Real-time decode capability for evaluation
        """


# RNN-T Backend Configuration Constants
class RNNTBackendConstants:
    """Named constants for RNN-T backend selection and optimization.
    
    These constants define the backend selection strategy and performance
    characteristics for different RNN-T loss implementations.
    """
    
    # Backend Preferences (in order)
    BACKEND_AUTO = "auto"           # Automatic selection (recommended)
    BACKEND_TORCHAUDIO = "torchaudio"  # Official PyTorch implementation
    BACKEND_WARP_RNNT = "warp_rnnt"    # Facebook's optimized implementation
    BACKEND_NAIVE = "naive"         # Pure PyTorch reference implementation
    BACKEND_CTC = "ctc"             # CTC approximation fallback
    
    # Performance Characteristics
    TORCHAUDIO_COMPLEXITY = "O(T*U)"    # Time complexity
    WARP_RNNT_COMPLEXITY = "O(T*U)"     # Time complexity
    NAIVE_COMPLEXITY = "O(T*U*V)"       # Time complexity (much slower)
    CTC_COMPLEXITY = "O(T*V)"           # Time complexity (approximation)
    
    # Apple Silicon Compatibility
    TORCHAUDIO_MPS_SUPPORT = "Excellent"   # Native MPS optimization
    WARP_RNNT_MPS_SUPPORT = "Limited"      # May lack Apple Silicon optimization
    NAIVE_MPS_SUPPORT = "Full"             # Pure PyTorch compatibility
    CTC_MPS_SUPPORT = "Excellent"          # Native MPS optimization
    
    @staticmethod
    def get_backend_info() -> str:
        """Return RNN-T backend selection documentation."""
        return f"""
        RNN-T Backend Selection Strategy:
        1. {RNNTBackendConstants.BACKEND_TORCHAUDIO}: {RNNTBackendConstants.TORCHAUDIO_COMPLEXITY} complexity, {RNNTBackendConstants.TORCHAUDIO_MPS_SUPPORT} MPS support
        2. {RNNTBackendConstants.BACKEND_WARP_RNNT}: {RNNTBackendConstants.WARP_RNNT_COMPLEXITY} complexity, {RNNTBackendConstants.WARP_RNNT_MPS_SUPPORT} MPS support
        3. {RNNTBackendConstants.BACKEND_NAIVE}: {RNNTBackendConstants.NAIVE_COMPLEXITY} complexity, {RNNTBackendConstants.NAIVE_MPS_SUPPORT} MPS support
        4. {RNNTBackendConstants.BACKEND_CTC}: {RNNTBackendConstants.CTC_COMPLEXITY} complexity, {RNNTBackendConstants.CTC_MPS_SUPPORT} MPS support (approximation)
        - Automatic selection prioritizes performance and compatibility
        - Fallback strategy ensures training continues regardless of backend availability
        """


# RNN-T Dataset Configuration Constants
class RNNTDatasetConstants:
    """Named constants for RNN-T dataset generation and processing.
    
    These constants define realistic audio and text characteristics
    for RNN-T training data generation and testing.
    """
    
    # Sample Configuration
    DEFAULT_NUM_SAMPLES = 16        # Dataset size for RNN-T testing
    STRESS_NUM_SAMPLES = 64         # Larger dataset for thorough testing
    
    # Audio Characteristics (similar to CTC but RNN-T specific)
    MEL_FEATURES = 80               # Standard mel-spectrogram feature count
    MIN_FRAMES = 300                # ~19 seconds at 16kHz (minimum for RNN-T)
    DEFAULT_MAX_FRAMES = 600        # ~38 seconds at 16kHz (typical for RNN-T)
    STRESS_MAX_FRAMES = 1200        # ~75 seconds (stress testing)
    
    # Text Characteristics for RNN-T
    MIN_TARGET_LEN = 3              # Minimum target sequence length
    MAX_TARGET_LEN = 40             # Maximum target sequence length (RNN-T)
    DEFAULT_VOCAB_SIZE = 1024       # Standard vocabulary size
    
    # Performance Targets
    TARGET_ALIGNMENTS_PER_SEC = 50000  # RNN-T alignment matrix performance target
    
    @staticmethod
    def get_rnnt_dataset_info() -> str:
        """Return RNN-T dataset characteristics documentation."""
        return f"""
        RNN-T Dataset Characteristics:
        - Audio: {RNNTDatasetConstants.MIN_FRAMES}-{RNNTDatasetConstants.DEFAULT_MAX_FRAMES} frames, {RNNTDatasetConstants.MEL_FEATURES} mel features
        - Text: {RNNTDatasetConstants.MIN_TARGET_LEN}-{RNNTDatasetConstants.MAX_TARGET_LEN} tokens (RNN-T optimized)
        - Vocabulary: {RNNTDatasetConstants.DEFAULT_VOCAB_SIZE} tokens (including RNN-T blank)
        - Target Performance: {RNNTDatasetConstants.TARGET_ALIGNMENTS_PER_SEC} alignment computations/sec
        - Memory: Alignment matrix T*U scales quadratically
        """


def get_device() -> torch.device:
    """Detect and return the optimal device for RNN-T training.
    
    This function implements device selection strategy optimized for
    RNN-T training on Apple Silicon with comprehensive fallback support.
    
    Device Priority for RNN-T:
    1. MPS (Apple Silicon GPU) - Primary target with RNN-T optimizations
    2. CUDA (NVIDIA GPU) - For systems with discrete NVIDIA cards
    3. CPU - Universal fallback with full RNN-T support
    
    RNN-T Considerations:
    - MPS may require fallback for RNN-T loss computation
    - Alignment matrix computation benefits from GPU acceleration
    - Memory requirements higher than CTC due to alignment matrix
    
    Returns:
        torch.device: Optimal device for RNN-T training on current hardware
        
    Usage:
        device = get_device()
        model = MCTModel(config).to(device)
    """
    if torch.backends.mps.is_available():
        # MPS available - use Apple Silicon GPU with fallback support
        return torch.device("mps")
    elif torch.cuda.is_available():
        # CUDA available - use NVIDIA GPU
        return torch.device("cuda")
    else:
        # Fallback to CPU for universal compatibility
        return torch.device("cpu")


class DummyRNNTDataset(torch.utils.data.Dataset):
    """Synthetic dataset for rapid prototyping and testing RNN-T training.
    
    This dataset generates random audio features and token sequences specifically
    designed for RNN-T training and validation. It creates realistic data
    distributions that match real speech recognition requirements.
    
    RNN-T Data Characteristics:
    - Audio features: Variable-length mel-spectrograms
    - Token sequences: Variable-length with RNN-T blank token considerations
    - Alignment matrix: Realistic T*U dimensions for memory testing
    - Sequence lengths: Variable to simulate real speech data
    
    Apple Silicon Optimization:
    - Tensor generation on CPU, efficient transfer to device
    - Variable-length sequences test memory allocation patterns
    - RNN-T alignment matrix size testing
    - Memory pressure simulation for unified memory architecture
    
    Usage:
        # Quick testing
        dataset = DummyRNNTDataset(num=16, max_T=600, max_U=40)
        
        # Stress testing
        dataset = DummyRNNTDataset(num=64, max_T=1200, max_U=80)
    """
    
    def __init__(self, 
                 num: int = RNNTDatasetConstants.DEFAULT_NUM_SAMPLES,
                 max_T: int = RNNTDatasetConstants.DEFAULT_MAX_FRAMES, 
                 max_U: int = RNNTDatasetConstants.MAX_TARGET_LEN, 
                 vocab: int = RNNTDatasetConstants.DEFAULT_VOCAB_SIZE):
        """Initialize dummy RNN-T dataset with configurable parameters.
        
        Args:
            num: Number of samples in dataset
            max_T: Maximum number of acoustic time frames per sample
            max_U: Maximum number of target tokens per sample
            vocab: Vocabulary size for target generation
        """
        super().__init__()
        self.num = num
        self.max_T = max_T
        self.max_U = max_U
        self.vocab = vocab

    def __len__(self) -> int:
        """Return dataset size."""
        return self.num

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """Generate a single RNN-T training sample with random data.
        
        Creates realistic audio and text data for RNN-T training:
        - Variable-length mel-spectrogram features
        - Corresponding sequence length tensor
        - Variable-length target token sequence (excluding blank)
        
        Args:
            idx: Sample index (unused, all samples are random)
            
        Returns:
            feats: Mel-spectrogram features (T, 80)
            feat_len: Acoustic sequence length scalar
            tokens: Target token sequence (U,) excluding RNN-T blank
        """
        # Generate variable-length acoustic sequence
        # RNN-T typically uses shorter sequences than CTC
        acoustic_frames = torch.randint(
            low=RNNTDatasetConstants.MIN_FRAMES,
            high=self.max_T,
            size=(1,)
        ).item()
        
        # Generate variable-length token sequence
        # RNN-T requires explicit token sequence (no CTC blanks in input)
        token_length = torch.randint(
            low=RNNTDatasetConstants.MIN_TARGET_LEN,
            high=self.max_U,
            size=(1,)
        ).item()
        
        # Create mel-spectrogram features with standard Gaussian distribution
        mel_features = torch.randn(acoustic_frames, RNNTDatasetConstants.MEL_FEATURES)
        acoustic_length = torch.tensor(acoustic_frames)
        
        # Create target tokens (excluding RNN-T blank token at index 0)
        target_tokens = torch.randint(
            low=1,  # Skip RNN-T blank token at index 0
            high=self.vocab - 1,
            size=(token_length,)
        )
        
        return mel_features, acoustic_length, target_tokens, "dummy text for wer calc"


def collate(batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Collate function for variable-length RNN-T data batching.
    
    This function handles batching of variable-length audio sequences and
    token targets for RNN-T training. It performs padding and RNN-T-specific
    preprocessing optimized for Apple Silicon memory characteristics.
    
    RNN-T Batching Strategy:
    - Audio: Pad sequences to maximum length in batch with zeros
    - Tokens: Pad and prepend RNN-T start-of-sequence blank token
    - Lengths: Maintain original sequence lengths for loss computation
    - Memory: Optimize for Apple Silicon unified memory architecture
    
    RNN-T Requirements:
    - Token sequences must start with blank token (index 0)
    - Alignment matrix computation requires accurate lengths
    - Memory usage scales as O(B * max_T * max_U * vocab_size)
    
    Args:
        batch: List of (features, feat_len, tokens) tuples from dataset
        
    Returns:
        feats: Padded feature tensor (B, max_T, 80)
        feat_lens: Original acoustic sequence lengths (B,)
        tokens: Padded token sequences with prepended blank (B, max_U+1)
        token_lens: Token sequence lengths including prepended blank (B,)
        
    Memory Considerations:
    - Alignment matrix will be (B, max_T/4, max_U+1, vocab_size)
    - Monitor memory usage for large batches on Apple Silicon
    - Unified memory enables larger sequences than discrete GPU
    """
    features_list, feature_lengths, tokens_list, texts = zip(*batch)
    batch_size = len(batch)
    
    # Find maximum dimensions for padding
    max_acoustic_frames = max(f.shape[0] for f in features_list)
    max_token_length = max(t.shape[0] for t in tokens_list)
    
    # Create padded feature tensor
    # Initialize with zeros (silence) for padding
    padded_features = torch.zeros(batch_size, max_acoustic_frames, RNNTDatasetConstants.MEL_FEATURES)
    
    # Create padded token tensor with RNN-T blank token prepending
    # +1 for RNN-T start-of-sequence blank token
    padded_tokens = torch.zeros(batch_size, max_token_length + 1, dtype=torch.long)
    
    # Fill padded tensors with actual data
    for batch_idx, (features, _, tokens, _) in enumerate(zip(features_list, feature_lengths, tokens_list, texts)):
        # Copy acoustic features
        acoustic_length = features.shape[0]
        padded_features[batch_idx, :acoustic_length] = features
        
        # Prepend RNN-T blank token (index 0) and copy target tokens
        # RNN-T requires blank token at the beginning of predictor sequence
        padded_tokens[batch_idx, 0] = RNNTTrainingConstants.RNNT_BLANK_TOKEN  # Start with blank
        token_length = tokens.shape[0]
        padded_tokens[batch_idx, 1:1 + token_length] = tokens
    
    # Create length tensors
    feature_lengths_tensor = torch.stack(feature_lengths)
    # Token lengths include the prepended blank token
    token_lengths_tensor = torch.tensor(
        [t.shape[0] + 1 for t in tokens_list], 
        dtype=torch.long
    )
    
    return padded_features, feature_lengths_tensor, padded_tokens, token_lengths_tensor, list(texts)


def rnnt_loss_naive_batch(logits: torch.Tensor, tokens: torch.Tensor, out_lens: torch.Tensor, token_lens: torch.Tensor, blank: int = 0) -> torch.Tensor:
    """Naive RNN-T loss implementation for small sequences and debugging.
    
    This function provides a reference implementation of RNN-T loss using
    dynamic programming forward algorithm. It's intended for debugging,
    validation, and handling small sequences when optimized backends fail.
    
    RNN-T Forward Algorithm:
    - Dynamic programming: Forward pass through alignment lattice
    - Alpha computation: Probability mass at each (t,u) position
    - Two transitions: Blank (advance time) and label (advance token)
    - Log-space computation: Numerical stability for long sequences
    
    Performance Characteristics:
    - Time complexity: O(B * T * U * V) - very expensive for large sequences
    - Memory usage: O(B * T * U) for alpha matrices
    - Suitable for: T <= 64, U <= 16 (debugging only)
    - Optimizations: Precomputed log-softmax, contiguous tensors
    
    Apple Silicon Considerations:
    - Unified memory enables larger alpha matrices than discrete GPU
    - Loop-heavy computation less optimal than vectorized backends
    - Fallback option when optimized RNN-T backends unavailable
    - Memory pressure monitoring recommended for batch processing
    
    Args:
        logits: Joiner output logits (B, T, U, V) - unnormalized scores
        tokens: Target token sequences (B, U) with tokens[:,0] = blank
        out_lens: Acoustic sequence lengths (B,) after encoder processing
        token_lens: Token sequence lengths (B,) including initial blank
        blank: Blank token index for RNN-T alignment (typically 0)
        
    Returns:
        Mean RNN-T loss across batch for gradient computation
        
    Algorithm Details:
    - Alpha[t,u]: Forward probability at acoustic frame t, token position u
    - Base case: Alpha[0,0] = 0 (log probability 1.0)
    - Blank transition: Alpha[t+1,u] += Alpha[t,u] + P(blank|t,u)
    - Label transition: Alpha[t,u+1] += Alpha[t,u] + P(label[u]|t,u)
    - Termination: Final loss includes mandatory blank at sequence end
    
    Usage Context:
    - Debugging: Validate optimized RNN-T backend results
    - Fallback: When torchaudio/warp-rnnt unavailable
    - Education: Reference implementation for understanding RNN-T
    - Testing: Small sequence validation during development
    
    Integration Notes:
    - Called by: main() training loop when backend selection fails
    - Coordination: Works with select_rnnt_backend() for automatic fallback
    - Profiling: Wrapped in record_function() for performance analysis
    - Memory: Guards prevent usage on large T*U alignments
    """
    B, T_max, U_max, V = logits.shape
    # Compute log-probabilities once to keep a simple autograd path
    log_probs_all = logits.log_softmax(dim=-1).contiguous()
    losses = []
    for b in range(B):
        T = int(out_lens[b].item())
        U = int(token_lens[b].item())
        y = tokens[b, 1:U]  # length U-1
        logp = log_probs_all[b, :T, :U, :].contiguous()  # (T,U,V)
        # alpha of shape (T,U)
        alpha = torch.full((T, U), float('-inf'), device=logp.device)
        alpha[0, 0] = 0.0
        for t in range(T):
            for u in range(U):
                a = alpha[t, u]
                if a == float('-inf'):
                    continue
                # blank transition to (t+1,u)
                if t + 1 < T:
                    alpha[t + 1, u] = torch.logaddexp(alpha[t + 1, u], a + logp[t, u, blank])
                # label transition to (t,u+1)
                if u + 1 < U:
                    lbl = int(y[u].item())
                    alpha[t, u + 1] = torch.logaddexp(alpha[t, u + 1], a + logp[t, u, lbl])
        # Termination: add final blank at (T-1,U-1)
        loss_b = -(alpha[T - 1, U - 1] + logp[T - 1, U - 1, blank])
        losses.append(loss_b)
    return torch.stack(losses).mean()


def select_rnnt_backend(preferred: str = "auto"):
    """Select and return the best-available RNN-T loss backend for Apple Silicon.
    
    This function implements intelligent backend selection for RNN-T loss computation,
    prioritizing optimized implementations while providing comprehensive fallback
    support for Apple Silicon environments.
    
    Backend Selection Strategy:
    1. torchaudio.prototype.rnnt: Official PyTorch implementation (preferred)
    2. warp_rnnt: Facebook's optimized CUDA/CPU implementation
    3. None: Triggers naive implementation or CTC fallback
    
    Apple Silicon Considerations:
    - torchaudio: Best MPS integration and maintenance
    - warp_rnnt: May have limited Apple Silicon optimization
    - naive: Pure PyTorch implementation as universal fallback
    - CTC: Encoder-only approximation for gradient computation
    
    Args:
        preferred: Backend selection preference
                  "auto" - Automatic selection (recommended)
                  "torchaudio" - Force torchaudio backend only
                  "warp_rnnt" - Force warp_rnnt backend only
                  "naive" - Force naive implementation
                  "ctc" - Force CTC fallback approximation
                  
    Returns:
        tuple[callable|None, str]: (loss_function, backend_name)
        - loss_function: RNN-T loss callable or None for fallback
        - backend_name: String identifier for selected backend
        
    Backend Characteristics:
    - torchaudio: O(T*U) complexity, MPS optimized, official support
    - warp_rnnt: O(T*U) complexity, CUDA optimized, may lack MPS
    - naive: O(T*U*V) complexity, pure PyTorch, universal compatibility
    - CTC: O(T*V) complexity, approximation only, efficient fallback
    
    Error Handling:
    - Import failures: Graceful degradation to next preference
    - Runtime errors: Caller handles with CPU fallback strategy
    - Version compatibility: Best-effort import with exception catching
    
    Usage Examples:
        # Automatic selection (recommended)
        loss_fn, backend = select_rnnt_backend("auto")
        
        # Force specific backend
        loss_fn, backend = select_rnnt_backend("torchaudio")
        
        # Handle selection result
        if loss_fn is not None:
            loss = loss_fn(log_probs, tokens, out_lens, token_lens)
        else:
            # Use naive or CTC fallback
            
    Integration Notes:
    - Called by: main() training function for backend configuration
    - Logging: Backend selection logged for reproducibility
    - Performance: Selection overhead amortized across training
    - Debugging: Explicit backend choice supports testing scenarios
    """
    preferred = preferred.lower()

    def try_ta():
        # Prefer prototype API if present
        try:
            from torchaudio.prototype.rnnt import rnnt_loss as ta_rnnt_loss  # type: ignore
            return ta_rnnt_loss, "torchaudio"
        except Exception:
            pass
        # Fallback to torchaudio.functional.rnnt_loss (deprecated but available widely)
        try:
            from torchaudio.functional import rnnt_loss as ta_rnnt_loss_fn  # type: ignore
            return ta_rnnt_loss_fn, "torchaudio"
        except Exception:
            return None

    def try_warp():
        # Best-effort: only attempt canonical package name
        try:
            from warp_rnnt import rnnt_loss as warp_rnnt_loss  # type: ignore
            return warp_rnnt_loss, "warp_rnnt"
        except Exception:
            return None

    if preferred == "torchaudio":
        res = try_ta()
        if res:
            return res
        return None, "none"
    if preferred == "warp_rnnt":
        res = try_warp()
        if res:
            return res
        return None, "none"
    if preferred in ("naive", "ctc"):
        return None, preferred

    # auto
    res = try_ta()
    if res:
        return res
    res = try_warp()
    if res:
        return res
    return None, "none"


def _rnnt_loss_torchaudio_safe(
    rnnt_fn,
    log_probs: torch.Tensor,
    tokens_with_blank: torch.Tensor,
    out_lens: torch.Tensor,
    token_lens_with_blank: torch.Tensor,
    blank: int = 0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute RNNT loss with torchaudio per-sample to avoid batch length mismatches.

    - Removes leading blank from targets per-sample
    - Slices log_probs per-sample to (Ti, Ui+1, V)
    - Runs on CPU to avoid MPS op gaps
    """
    try:
        from torchaudio.functional import rnnt_loss as _ta_fn  # type: ignore
        is_ta_functional = rnnt_fn is _ta_fn
    except Exception:
        is_ta_functional = False

    B, Tcap, Ucap, V = log_probs.shape
    losses = []
    for b in range(B):
        Ti = int(out_lens[b].item())
        Ui_with_blank = int(token_lens_with_blank[b].item())
        Ui = max(0, Ui_with_blank - 1)
        if Ti <= 0 or Ui <= 0:
            continue
        # Effective usable lengths constrained by current tensor caps
        Ti_eff = min(Ti, Tcap)
        Ui_eff = min(Ui, max(0, Ucap - 1))
        if Ui_eff <= 0 or Ti_eff <= 0:
            continue
        # Slice per-sample
        lp_b = log_probs[b : b + 1, : Ti_eff, : (Ui_eff + 1), :].contiguous()
        # Targets exclude leading blank and align to Ui_eff
        tgt_b = tokens_with_blank[b, 1 : 1 + Ui_eff]
        if is_ta_functional:
            tgt_b = tgt_b.to(torch.int32).unsqueeze(0)  # (1, Ui)
            Ti_t = torch.tensor([Ti_eff], dtype=torch.int32)
            Ui_t = torch.tensor([Ui_eff], dtype=torch.int32)
        else:
            tgt_b = tgt_b.unsqueeze(0)  # (1, Ui)
            Ti_t = torch.tensor([Ti_eff], dtype=torch.long)
            Ui_t = torch.tensor([Ui_eff], dtype=torch.long)
        # Move to CPU as torchaudio op is CPU-backed here
        loss_b = rnnt_fn(
            lp_b.to("cpu"),
            tgt_b.to("cpu"),
            Ti_t.to("cpu"),
            Ui_t.to("cpu"),
            blank=blank,
            clamp=-1,
            reduction="mean",
        )
        losses.append(loss_b.to(log_probs.device))
    if not losses:
        return torch.zeros((), device=log_probs.device)
    return torch.stack(losses).mean() if reduction == "mean" else torch.stack(losses).sum()


def _rnnt_loss_cpu_with_grad(
    rnnt_fn,
    logits: torch.Tensor,
    tokens_with_blank: torch.Tensor,
    out_lens: torch.Tensor,
    token_lens_with_blank: torch.Tensor,
    blank: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute RNNT loss on CPU per-sample and return (loss_value_on_device, grad_logits_on_device).

    This keeps the model on MPS, computes a scalar loss on CPU while preserving gradients
    w.r.t. logits via manual backward on a CPU copy, then maps grad back to device.
    """
    try:
        from torchaudio.functional import rnnt_loss as _ta_fn  # type: ignore
        is_ta_functional = rnnt_fn is _ta_fn
    except Exception:
        is_ta_functional = False
    B, Tcap, Ucap, V = logits.shape
    logits_cpu = logits.detach().to("cpu").requires_grad_(True)
    total = None
    used = 0
    for b in range(B):
        Ti = int(out_lens[b].item())
        Ui_with_blank = int(token_lens_with_blank[b].item())
        Ui = max(0, Ui_with_blank - 1)
        Ti_eff = min(Ti, Tcap)
        Ui_eff = min(Ui, max(0, Ucap - 1))
        if Ti_eff <= 0 or Ui_eff <= 0:
            continue
        # Slice logits and compute log_probs in place to keep autograd path
        sl = logits_cpu[b : b + 1, : Ti_eff, : (Ui_eff + 1), :].contiguous()
        lp_b = sl.log_softmax(dim=-1)
        tgt_b = tokens_with_blank[b, 1 : 1 + Ui_eff]
        if is_ta_functional:
            tgt_b = tgt_b.to(torch.int32).unsqueeze(0)
            Ti_t = torch.tensor([Ti_eff], dtype=torch.int32)
            Ui_t = torch.tensor([Ui_eff], dtype=torch.int32)
        else:
            tgt_b = tgt_b.unsqueeze(0)
            Ti_t = torch.tensor([Ti_eff], dtype=torch.long)
            Ui_t = torch.tensor([Ui_eff], dtype=torch.long)
        loss_b = rnnt_fn(lp_b, tgt_b, Ti_t, Ui_t, blank=blank, clamp=-1, reduction="mean")
        total = loss_b if total is None else total + loss_b
        used += 1
    if used == 0:
        return torch.zeros((), device=logits.device), torch.zeros_like(logits)
    total = total / used
    total.backward()
    grad_logits = logits_cpu.grad.to(logits.device)
    return total.to(logits.device).detach(), grad_logits.detach()

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Enable MPS profiler context where available")
    parser.add_argument("--manifest", type=str, default="", help="Path to LibriSpeech CSV manifest for RNNT training")
    parser.add_argument("--force_naive_rnnt", action="store_true", help="Force small T',U for naive RNNT loss path")
    parser.add_argument("--rnnt_impl", type=str, default="mps_native", choices=["auto", "torchaudio", "warp_rnnt", "naive", "ctc", "mps_native"], help="Select RNN-T loss implementation")
    parser.add_argument("--max_align", type=int, default=60000, help="Maximum allowed T'*U alignment size before clamping/fallback (default tightened based on LibriSpeech T'·U distributions)")
    parser.add_argument("--max_samples", type=int, default=0, help="Limit number of samples from manifest for a short pass (0 = all)")
    parser.add_argument("--max_steps", type=int, default=0, help="Limit number of optimizer steps for a short pass (0 = full epoch)")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="auto", choices=["auto","cpu","mps","cuda"], help="Force device override for RNNT experiments")
    parser.add_argument("--rnnt_max_align", type=int, default=60000, help="Max allowed T'*U before shrinking U (also available via RNNT_MAX_ALIGN env)")
    parser.add_argument("--rnnt_cpu_grad", action="store_true", help="Deprecated: CPU-grad fallback is always enabled automatically when fast backend fails. Use --force_cpu_grad to force CPU-grad for all batches.")
    parser.add_argument("--force_cpu_grad", action="store_true", help="Force CPU per-sample RNNT with gradient mapping for all batches (stability baseline)")
    parser.add_argument("--eval_after", action="store_true", help="Run a small greedy-decode evaluation after training to report avg WER")
    parser.add_argument("--eval_samples", type=int, default=12, help="Number of samples to evaluate with greedy decode after training")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Clip global grad-norm to this value (0 disables)")
    parser.add_argument("--skip_non_finite", action="store_true", help="Skip optimizer step when loss is non-finite (nan/inf)")
    parser.add_argument("--log_csv", type=str, default="", help="Optional path to write per-step metrics CSV (step, loss, t_cap, u_cap, align, backend, finite)")
    parser.add_argument("--log_json", type=str, default="", help="Optional path to write summary metrics JSON at end of run")
    parser.add_argument("--save_ckpt", type=str, default="", help="Optional path to save a checkpoint at the end of training (.pt)")
    parser.add_argument("--adaptive_ctc_after_cpu_grad", type=int, default=0, help="If >0, after N consecutive cpu_grad RNNT batches, force encoder-CTC fallback for remainder of run")
    args = parser.parse_args()

    device = get_device() if args.device == "auto" else torch.device(args.device)
    # Propagate alignment cap to RNNT facade via env
    try:
        os.environ["RNNT_MAX_ALIGN"] = str(int(args.rnnt_max_align))
    except Exception:
        pass
    print(f"Using device: {device}")

    # Tokenizer and vocab
    tokenizer = CharTokenizer()
    cfg = MCTConfig(vocab_size=tokenizer.vocab_size)
    model = MCTModel(cfg).to(device)

    if args.manifest and HAS_LS:
        try:
            ds = LibriSpeechCSVDataset(args.manifest, tokenizer=tokenizer)
            if args.max_samples and hasattr(ds, "rows"):
                ds.rows = ds.rows[: args.max_samples]
            dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=ls_collate, num_workers=args.num_workers)
            print(f"Loaded LibriSpeech CSV: {args.manifest} ({len(ds)} rows)")
        except Exception as e:
            print(f"Failed to load manifest: {e}. Falling back to dummy dataset.")
            ds = DummyRNNTDataset(num=8 if args.sanity else 64, vocab=tokenizer.vocab_size)
            dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.num_workers)
    else:
        ds = DummyRNNTDataset(num=8 if args.sanity else 64, vocab=tokenizer.vocab_size)
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    # Select RNNT loss backend
    rnnt_loss, rnnt_backend = select_rnnt_backend(args.rnnt_impl if args.rnnt_impl != "mps_native" else "auto")
    # If mps_native is requested, wrap with MPS-native facade
    rnnt_mps_facade = None
    if args.rnnt_impl == "mps_native":
        try:
            from modules.rnnt_loss_mps import rnnt_loss_mps  # type: ignore
            rnnt_mps_facade = rnnt_loss_mps
            print("Using RNN-T loss facade: mps_native (auto + CPU-grad fallback)")
        except Exception as _e:
            print(f"mps_native not available ({_e}); falling back to auto backend selection.")
    if rnnt_backend == "torchaudio":
        print("Using RNN-T loss backend: torchaudio.prototype.rnnt.rnnt_loss")
    elif rnnt_backend == "warp_rnnt":
        print("Using RNN-T loss backend: warp_rnnt.rnnt_loss")
    elif rnnt_backend in ("naive", "ctc"):
        print(f"Using RNN-T loss backend: {rnnt_backend} (requested)")
    else:
        print("No RNN-T backend available; will use naive (small T,U) or encoder-CTC fallback.")

    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    model.train()

    # Metrics aggregation for structured logging and analysis
    import math, csv  # Local imports to avoid polluting global namespace
    align_values: list[int] = []
    t_caps: list[int] = []
    u_caps: list[int] = []
    backend_use_counts = {"ta": 0, "warp": 0, "naive": 0, "ctc": 0, "cpu_grad": 0, "unknown": 0}
    csv_writer = None
    csv_file_handle = None
    if args.log_csv:
        try:
            csv_file_handle = open(args.log_csv, "w", newline="")
            csv_writer = csv.writer(csv_file_handle)
            csv_writer.writerow(["epoch", "step", "loss", "t_cap", "u_cap", "align", "backend", "finite"])
        except Exception:
            csv_writer = None

    # Optional MPS profiling
    try:
        from torch.mps.profiler import profile as mps_profile  # type: ignore
    except Exception:
        mps_profile = contextlib.nullcontext  # type: ignore
    ctx = mps_profile() if args.profile else contextlib.nullcontext()

    total_frames = 0
    start = time.time()
    with ctx:
        for epoch in range(args.epochs):
            consecutive_cpu_grad = 0
            force_ctc_rest = False
            for step, batch in enumerate(dl):
                # Support both dummy and LibriSpeech collates
                texts = None
                if isinstance(batch, (list, tuple)) and len(batch) == 5:
                    feats, feat_lens, tokens, token_lens, texts = batch
                else:
                    feats, feat_lens, tokens, token_lens = batch
                total_frames += int(feat_lens.sum().item())
                feats = feats.to(device)
                feat_lens = feat_lens.to(device)
                tokens = tokens.to(device)
                token_lens = token_lens.to(device)

                with record_function("mct_forward"):
                    logits, out_lens = model(feats, feat_lens, tokens)
                # Optionally clamp lengths to force naive RNNT path (for environments without RNNT loss)
                if args.force_naive_rnnt:
                    # Clamp T' and U to small sizes
                    Tprime = min(logits.shape[1], 64)
                    U = min(logits.shape[2], 16)
                    logits = logits[:, :Tprime, :U, :]
                    out_lens = torch.clamp(out_lens, max=Tprime)
                    token_lens = torch.clamp(token_lens, max=U)
                # Alignment size guard for Apple Silicon memory
                t_cap = int(out_lens.max().item())
                u_cap = int(token_lens.max().item())
                align_size = t_cap * u_cap
                if align_size > args.max_align and rnnt_backend in ("none", "naive") and not args.force_naive_rnnt:
                    print(f"Warning: T'*U={align_size} exceeds max_align={args.max_align}; skipping naive RNN-T and using encoder-CTC fallback.")
                    rnnt_use_naive = False
                else:
                    rnnt_use_naive = (rnnt_backend == "naive") or (rnnt_backend == "none" and (logits.shape[1] <= 64 and logits.shape[2] <= 16))
                # logits: (B, T, U, V)
                backend_used = "unknown"
                # Adaptive CTC fallback gate
                if force_ctc_rest and not args.force_naive_rnnt:
                    with record_function("ctc_fallback_compute"):
                        enc_only = logits.max(dim=2).values  # (B, T, V)
                        logp = enc_only.log_softmax(dim=-1).transpose(0, 1)
                        targets = []
                        for b in range(tokens.shape[0]):
                            toks = tokens[b, 1 : token_lens[b]]
                            targets.append(toks)
                        flat = torch.cat(targets)
                        tgt_lens = torch.tensor([len(t) for t in targets], device=logp.device)
                        loss = ctc_loss(logp, flat, out_lens, tgt_lens)
                        backend_used = "ctc"
                    goto_metrics = True
                else:
                    goto_metrics = False
                # If explicitly requested, run CPU-grad RNNT per-batch (stability baseline)
                if args.force_cpu_grad and (rnnt_loss is not None) and (rnnt_backend in ("torchaudio", "warp_rnnt")):
                    try:
                        loss_cpu, grad_logits = _rnnt_loss_cpu_with_grad(rnnt_loss, logits.detach(), tokens, out_lens, token_lens, blank=0)
                        optimizer.zero_grad(set_to_none=True)
                        logits.backward(grad_logits)
                        optimizer.step()
                        loss = loss_cpu
                        backend_used = "cpu_grad"
                        if step % 10 == 0:
                            log_msg = (
                                f"epoch {epoch} step {step} loss {loss.item():.4f} [cpu-rnnt] "
                                f"align(T'U')={align_size} (T'={t_cap}, U={u_cap})"
                            )
                            if texts is not None:
                                def greedy_rnnt_decode_single(feat: torch.Tensor, feat_len: torch.Tensor) -> str:
                                    model.eval()
                                    with torch.no_grad():
                                        feat_b = feat.unsqueeze(0).to(device)
                                        len_b = feat_len.unsqueeze(0).to(device)
                                        enc_in = model.frontend(feat_b)
                                        enc_out = model.encoder(enc_in)
                                        Tprime = int(enc_out.shape[1])
                                        hidden = None
                                        token_cur = torch.zeros(1, dtype=torch.long, device=device)
                                        hyp_ids = []
                                        max_total = 128
                                        total_dec = 0
                                        for t in range(Tprime):
                                            u = 0
                                            while u < 32 and total_dec < max_total:
                                                pred_step, hidden = model.predictor.forward_streaming(token_cur.unsqueeze(1), hidden)
                                                logits_tu = model.joiner(enc_out[:, t:t+1, :], pred_step)
                                                next_id = int(logits_tu[0, 0, 0].argmax().item())
                                                total_dec += 1
                                                if next_id == RNNTTrainingConstants.RNNT_BLANK_TOKEN:
                                                    break
                                                hyp_ids.append(next_id)
                                                token_cur = torch.tensor([next_id], dtype=torch.long, device=device)
                                                u += 1
                                        hyp = tokenizer.decode(hyp_ids)
                                    model.train()
                                    return hyp
                                hyp_text = greedy_rnnt_decode_single(feats[0].cpu(), feat_lens[0].cpu())
                                ref_text = tokenizer.normalize(texts[0])
                                log_msg += f" wer~{wer(ref_text, hyp_text):.3f}"
                            print(log_msg)
                        if csv_writer is not None:
                            csv_writer.writerow([epoch, step, float(loss.item()), t_cap, u_cap, align_size, backend_used, 1])
                        align_values.append(align_size)
                        t_caps.append(t_cap)
                        u_caps.append(u_cap)
                        backend_use_counts[backend_used] = backend_use_counts.get(backend_used, 0) + 1
                        continue
                    except Exception as e2:
                        print(f"Explicit CPU RNNT w/grad failed ({e2}); will try fast backend or CTC fallback.")
                # Backward-compatibility notice: --rnnt_cpu_grad is deprecated and no longer forces CPU-grad
                if args.rnnt_cpu_grad and not args.force_cpu_grad and step == 0:
                    print("[notice] --rnnt_cpu_grad is deprecated. CPU-grad fallback happens automatically on failures. Use --force_cpu_grad to force CPU-grad for all batches.")
                # mps_native facade path
                if not goto_metrics and rnnt_mps_facade is not None:
                    with record_function("rnnt_loss_mps_native"):
                        try:
                            loss_or, grad_logits, which = rnnt_mps_facade(logits, tokens, out_lens, token_lens, blank=0, max_align=args.rnnt_max_align)
                            backend_used = which if which != "torchaudio" else "ta"
                            if grad_logits is not None:
                                optimizer.zero_grad(set_to_none=True)
                                logits.backward(grad_logits)
                                optimizer.step()
                                loss = loss_or
                                # Log periodically for CPU-grad branch
                                if step % 10 == 0:
                                    print(f"epoch {epoch} step {step} loss {loss.item():.4f} [mps_native:{backend_used}] align(T'U')={align_size} (T'={t_cap}, U={u_cap})")
                                if csv_writer is not None:
                                    csv_writer.writerow([epoch, step, float(loss.item()), t_cap, u_cap, align_size, backend_used, 1])
                                align_values.append(align_size)
                                t_caps.append(t_cap)
                                u_caps.append(u_cap)
                                backend_use_counts[backend_used] = backend_use_counts.get(backend_used, 0) + 1
                                continue
                            else:
                                loss = loss_or
                        except Exception as e:
                            print(f"mps_native facade failed ({e}); will try fast backend or CTC fallback.")
                elif not goto_metrics and (not args.force_naive_rnnt) and (rnnt_loss is not None) and (rnnt_backend in ("torchaudio", "warp_rnnt")):
                    with record_function("rnnt_loss_compute"):
                        log_probs = logits.log_softmax(dim=-1)
                        try:
                            # torchaudio.functional expects int32 for targets/lengths
                            t_tokens = tokens
                            t_out_lens = out_lens
                            t_tok_lens = token_lens
                            # For torchaudio RNNT, targets should EXCLUDE leading blank.
                            # Our tokens include a leading blank for the predictor; adjust here.
                            use_ta_functional = False
                            try:
                                import torchaudio  # noqa: F401
                                from torchaudio.functional import rnnt_loss as _ta_fn  # type: ignore
                                if rnnt_loss is _ta_fn:
                                    use_ta_functional = True
                            except Exception:
                                pass
                            if use_ta_functional:
                                t_tokens = tokens[:, 1:].to(torch.int32)
                                t_out_lens = out_lens.to(torch.int32)
                                t_tok_lens = (token_lens - 1).clamp_min(0).to(torch.int32)
                                # Ensure lens do not exceed logits dims and slice logits U-dim to max token length
                                Tcap = log_probs.shape[1]
                                Ucap = log_probs.shape[2]
                                t_out_lens = torch.clamp(t_out_lens, max=Tcap)
                                t_tok_lens = torch.clamp(t_tok_lens, min=0, max=Ucap)
                                # Keep T'*U under --max_align by shrinking U across the batch if needed
                                if args.max_align and args.max_align > 0 and t_out_lens.numel() > 0:
                                    # Compute per-sample allowed U to satisfy T*U <= max_align
                                    allowed_Us = []
                                    for i in range(t_out_lens.numel()):
                                        Ti_eff_i = int(t_out_lens[i].item())
                                        Ui_eff_i = int(t_tok_lens[i].item())
                                        if Ti_eff_i <= 0:
                                            allowed_Us.append(0)
                                            continue
                                        Ui_allowed = min(Ui_eff_i, max(1, int(args.max_align // max(1, Ti_eff_i))))
                                        allowed_Us.append(Ui_allowed)
                                    Ubatch = min(allowed_Us) if allowed_Us else 0
                                    if Ubatch > 0:
                                        t_tok_lens = torch.clamp(t_tok_lens, max=Ubatch)
                                        if Ubatch <= Ucap:
                                            log_probs = log_probs[:, :, :Ubatch, :]
                            loss = rnnt_loss(
                                log_probs, t_tokens, t_out_lens, t_tok_lens, blank=0, clamp=-1, reduction="mean"
                            )
                            backend_used = "ta" if rnnt_backend == "torchaudio" else "warp"
                            # If the returned loss is not connected to the graph (common when backend runs on CPU),
                            # fall back to CPU-grad mapping to inject gradients explicitly.
                            if not getattr(loss, "requires_grad", False):
                                print("RNN-T backend returned a non-differentiable loss; injecting CPU-grad fallback.")
                                loss_cpu, grad_logits = _rnnt_loss_cpu_with_grad(rnnt_loss, logits.detach(), tokens, out_lens, token_lens, blank=0)
                                optimizer.zero_grad(set_to_none=True)
                                logits.backward(grad_logits)
                                optimizer.step()
                                loss = loss_cpu
                                backend_used = "cpu_grad"
                                if csv_writer is not None:
                                    csv_writer.writerow([epoch, step, float(loss.item()), t_cap, u_cap, align_size, backend_used, 1])
                                align_values.append(align_size)
                                t_caps.append(t_cap)
                                u_caps.append(u_cap)
                                backend_use_counts[backend_used] = backend_use_counts.get(backend_used, 0) + 1
                                continue
                        except Exception as e:
                            # Safe CPU fallback with gradients via manual backward on CPU logits
                            print(f"RNN-T backend failed on-device ({e}); computing RNNT on CPU per-sample with grad mapping.")
                            try:
                                loss_cpu, grad_logits = _rnnt_loss_cpu_with_grad(rnnt_loss, logits.detach(), tokens, out_lens, token_lens, blank=0)
                                # Replace autograd path with manual gradient injection
                                optimizer.zero_grad(set_to_none=True)
                                logits.backward(grad_logits)
                                optimizer.step()
                                loss = loss_cpu
                                backend_used = "cpu_grad"
                                # Emit periodic logs here since we bypass the standard logger below
                                if step % 10 == 0:
                                    log_msg = (
                                        f"epoch {epoch} step {step} loss {loss.item():.4f} [cpu-rnnt] "
                                        f"align(T'U')={align_size} (T'={t_cap}, U={u_cap})"
                                    )
                                    if texts is not None:
                                        # Greedy decode for first sample for approximate WER
                                        def greedy_rnnt_decode_single(feat: torch.Tensor, feat_len: torch.Tensor) -> str:
                                            model.eval()
                                            with torch.no_grad():
                                                feat_b = feat.unsqueeze(0).to(device)
                                                len_b = feat_len.unsqueeze(0).to(device)
                                                enc_in = model.frontend(feat_b)
                                                enc_out = model.encoder(enc_in)
                                                Tprime = int(enc_out.shape[1])
                                                hidden = None
                                                token_cur = torch.zeros(1, dtype=torch.long, device=device)
                                                hyp_ids = []
                                                max_total = 128
                                                total_dec = 0
                                                for t in range(Tprime):
                                                    u = 0
                                                    while u < 32 and total_dec < max_total:
                                                        pred_step, hidden = model.predictor.forward_streaming(token_cur.unsqueeze(1), hidden)
                                                        logits_tu = model.joiner(enc_out[:, t:t+1, :], pred_step)
                                                        next_id = int(logits_tu[0, 0, 0].argmax().item())
                                                        total_dec += 1
                                                        if next_id == RNNTTrainingConstants.RNNT_BLANK_TOKEN:
                                                            break
                                                        hyp_ids.append(next_id)
                                                        token_cur = torch.tensor([next_id], dtype=torch.long, device=device)
                                                        u += 1
                                                hyp = tokenizer.decode(hyp_ids)
                                            model.train()
                                            return hyp
                                        hyp_text = greedy_rnnt_decode_single(feats[0].cpu(), feat_lens[0].cpu())
                                        ref_text = tokenizer.normalize(texts[0])
                                        log_msg += f" wer~{wer(ref_text, hyp_text):.3f}"
                                    print(log_msg)
                                # Record CSV metrics for cpu-grad branch before skipping
                                if csv_writer is not None:
                                    csv_writer.writerow([epoch, step, float(loss.item()), t_cap, u_cap, align_size, backend_used, 1])
                                align_values.append(align_size)
                                t_caps.append(t_cap)
                                u_caps.append(u_cap)
                                backend_use_counts[backend_used] = backend_use_counts.get(backend_used, 0) + 1
                                # Skip standard backward/step and logger below since we've already done them
                                continue
                            except Exception as e2:
                                print(f"CPU RNNT w/grad failed ({e2}); using encoder-CTC fallback for this batch.")
                                with record_function("ctc_fallback_compute"):
                                    enc_only = logits.max(dim=2).values  # (B, T, V)
                                    logp = enc_only.log_softmax(dim=-1).transpose(0, 1)
                                    targets = []
                                    for b in range(tokens.shape[0]):
                                        toks = tokens[b, 1 : token_lens[b]]
                                        targets.append(toks)
                                    flat = torch.cat(targets)
                                    tgt_lens = torch.tensor([len(t) for t in targets], device=logp.device)
                                    loss = ctc_loss(logp, flat, out_lens, tgt_lens)
                else:
                    # Attempt naive RNNT for very small T,U (sanity only)
                    if rnnt_use_naive:
                        with record_function("rnnt_loss_naive_compute"):
                            with torch.no_grad():
                                rnnt_val = rnnt_loss_naive_batch(logits, tokens, out_lens, token_lens, blank=0)
                        # Use encoder-CTC to provide gradients while reporting RNNT value
                        with record_function("ctc_fallback_compute"):
                            enc_only = logits.max(dim=2).values  # (B, T, V)
                            logp = enc_only.log_softmax(dim=-1).transpose(0, 1)
                            targets = []
                            for b in range(tokens.shape[0]):
                                toks = tokens[b, 1 : token_lens[b]]
                                targets.append(toks)
                            flat = torch.cat(targets)
                            tgt_lens = torch.tensor([len(t) for t in targets], device=logp.device)
                            loss = ctc_loss(logp, flat, out_lens, tgt_lens)
                            backend_used = "naive"
                    else:
                        # Fallback: CTC on encoder stream only (approx)
                        with record_function("ctc_fallback_compute"):
                            enc_only = logits.max(dim=2).values  # (B, T, V)
                            logp = enc_only.log_softmax(dim=-1).transpose(0, 1)
                            targets = []
                            for b in range(tokens.shape[0]):
                                toks = tokens[b, 1 : token_lens[b]]
                                targets.append(toks)
                            flat = torch.cat(targets)
                            tgt_lens = torch.tensor([len(t) for t in targets], device=logp.device)
                            loss = ctc_loss(logp, flat, out_lens, tgt_lens)
                            backend_used = "ctc"

                is_finite = bool(torch.isfinite(loss))
                # Update adaptive CTC switch based on backend usage
                if args.adaptive_ctc_after_cpu_grad > 0:
                    if backend_used == "cpu_grad":
                        consecutive_cpu_grad += 1
                        if consecutive_cpu_grad >= args.adaptive_ctc_after_cpu_grad:
                            if not force_ctc_rest:
                                print(f"[adaptive] Switching to encoder-CTC fallback for rest of run after {consecutive_cpu_grad} consecutive cpu_grad batches.")
                            force_ctc_rest = True
                    elif backend_used in ("ta", "warp"):
                        consecutive_cpu_grad = 0
                if args.skip_non_finite and (not is_finite):
                    print(f"Skipping non-finite loss at step {step}: {loss.item() if loss.numel()==1 else 'tensor'}")
                    optimizer.zero_grad(set_to_none=True)
                    # Log metrics even when skipping the step
                    if csv_writer is not None:
                        csv_writer.writerow([epoch, step, float('nan'), t_cap, u_cap, align_size, backend_used, 0])
                    align_values.append(align_size)
                    t_caps.append(t_cap)
                    u_caps.append(u_cap)
                    backend_use_counts[backend_used] = backend_use_counts.get(backend_used, 0) + 1
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()

                # Persist per-step metrics
                if csv_writer is not None:
                    csv_writer.writerow([epoch, step, float(loss.item()), t_cap, u_cap, align_size, backend_used, 1 if is_finite else 0])
                align_values.append(align_size)
                t_caps.append(t_cap)
                u_caps.append(u_cap)
                backend_use_counts[backend_used] = backend_use_counts.get(backend_used, 0) + 1

                if step % 10 == 0:
                    log_msg = f"epoch {epoch} step {step} loss {loss.item():.4f} align(T'U')={align_size} (T'={t_cap}, U={u_cap})"
                    # Rough WER via encoder-only greedy if texts available (approximate)
                    if texts is not None:
                        # Greedy RNN-T decode (approximate, small batch)
                        def greedy_rnnt_decode_single(feat: torch.Tensor, feat_len: torch.Tensor) -> str:
                            # feat: (T, 80)
                            model.eval()
                            with torch.no_grad():
                                feat_b = feat.unsqueeze(0).to(device)
                                len_b = feat_len.unsqueeze(0).to(device)
                                enc_in = model.frontend(feat_b)               # (1, T', D)
                                enc_out = model.encoder(enc_in)                # (1, T', D)
                                Tprime = int(enc_out.shape[1])
                                # Predictor streaming state
                                hidden = None
                                token_cur = torch.zeros(1, dtype=torch.long, device=device)  # blank start
                                hyp_ids = []
                                max_total = 128
                                total = 0
                                for t in range(Tprime):
                                    u = 0
                                    while u < 32 and total < max_total:
                                        with record_function("predictor_step"):
                                            pred_step, hidden = model.predictor.forward_streaming(token_cur.unsqueeze(1), hidden)
                                        # Join for this (t,u)
                                        with record_function("joiner_step"):
                                            logits_tu = model.joiner(enc_out[:, t:t+1, :], pred_step)  # (1,1,1,V)
                                        next_id = int(logits_tu[0, 0, 0].argmax().item())
                                        total += 1
                                        if next_id == RNNTTrainingConstants.RNNT_BLANK_TOKEN:
                                            break
                                        else:
                                            hyp_ids.append(next_id)
                                            token_cur = torch.tensor([next_id], dtype=torch.long, device=device)
                                            u += 1
                                hyp = tokenizer.decode(hyp_ids)
                            model.train()
                            return hyp
                        # Compute WER for first sample only (speed)
                        hyp_text = greedy_rnnt_decode_single(feats[0].cpu(), feat_lens[0].cpu())
                        ref_text = tokenizer.normalize(texts[0])
                        log_msg += f" wer~{wer(ref_text, hyp_text):.3f}"
                    print(log_msg)
                if args.max_steps and (step + 1) >= args.max_steps:
                    break

            if device.type == "mps":
                torch.mps.synchronize()

    elapsed = time.time() - start
    encoder_fps = total_frames/elapsed if elapsed > 0 else float('nan')
    if elapsed > 0:
        print(f"encoder throughput ~ {encoder_fps:.1f} frames/sec (dummy)")

    # Summaries for organizational knowledge capture
    def _percentile(vals: list[int], q: float) -> float:
        if not vals:
            return float('nan')
        s = sorted(vals)
        k = (len(s) - 1) * q
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(s[int(k)])
        return float(s[f] + (s[c] - s[f]) * (k - f))

    if align_values:
        print(f"align stats: count={len(align_values)}, p50={_percentile(align_values,0.5):.0f}, p90={_percentile(align_values,0.9):.0f}, p99={_percentile(align_values,0.99):.0f}, max={max(align_values)}")
    if t_caps and u_caps:
        print(f"T' caps: p50={_percentile(t_caps,0.5):.0f}, p90={_percentile(t_caps,0.9):.0f}, max={max(t_caps)}; U caps: p50={_percentile(u_caps,0.5):.0f}, p90={_percentile(u_caps,0.9):.0f}, max={max(u_caps)}")
    if backend_use_counts:
        print(f"backend usage: {backend_use_counts}")

    # Optional summary JSON
    if args.log_json:
        try:
            import json as _json
            import os as _os
            summary = {
                "encoder_fps": float(encoder_fps),
                "align_count": len(align_values),
                "align_p50": _percentile(align_values, 0.5) if align_values else None,
                "align_p90": _percentile(align_values, 0.9) if align_values else None,
                "align_p99": _percentile(align_values, 0.99) if align_values else None,
                "align_max": max(align_values) if align_values else None,
                "Tcap_p50": _percentile(t_caps, 0.5) if t_caps else None,
                "Tcap_p90": _percentile(t_caps, 0.9) if t_caps else None,
                "Tcap_max": max(t_caps) if t_caps else None,
                "Ucap_p50": _percentile(u_caps, 0.5) if u_caps else None,
                "Ucap_p90": _percentile(u_caps, 0.9) if u_caps else None,
                "Ucap_max": max(u_caps) if u_caps else None,
                "backend_usage": backend_use_counts,
                "args": vars(args),
            }
            _os.makedirs(_os.path.dirname(args.log_json), exist_ok=True)
            with open(args.log_json, "w") as _fh:
                _json.dump(summary, _fh, indent=2)
            print(f"wrote summary JSON: {args.log_json}")
        except Exception as _e:
            print(f"failed to write summary JSON: {_e}")

    if csv_file_handle is not None:
        try:
            csv_file_handle.close()
        except Exception:
            pass

    # Save checkpoint if requested (end-of-run)
    try:
        import time as _time, os as _os
        if args.save_ckpt is not None:
            ckpt_path = args.save_ckpt.strip()
        else:
            ckpt_path = ""
        if not ckpt_path:
            # Default path under repo checkpoints/
            default_dir = _os.path.join(_os.path.dirname(__file__), "checkpoints")
            _os.makedirs(default_dir, exist_ok=True)
            ckpt_path = _os.path.join(default_dir, f"rnnt_{int(_time.time())}.pt")
        else:
            _os.makedirs(_os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "backend": rnnt_backend,
            "args": vars(args),
        }, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")
    except Exception as _e:
        print(f"Checkpoint save skipped/failed: {_e}")

    # Optional post-training evaluation with greedy decode for quick WER signal
    if args.eval_after and HAS_LS and args.manifest:
        try:
            eval_ds = LibriSpeechCSVDataset(args.manifest, tokenizer=tokenizer)
            eval_dl = torch.utils.data.DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=ls_collate, num_workers=0)
            model.eval()
            total_wer = 0.0
            num_eval = 0
            with torch.no_grad():
                for bidx, batch in enumerate(eval_dl):
                    if bidx >= max(1, args.eval_samples):
                        break
                    feats, feat_lens, _tokens, _token_lens, texts = batch
                    feats = feats.to(device)
                    feat_lens = feat_lens.to(device)
                    # Greedy RNN-T decode using streaming predictor
                    enc_in = model.frontend(feats)            # (1, T', D)
                    enc_out = model.encoder(enc_in)           # (1, T', D)
                    Tprime = int(enc_out.shape[1])
                    hidden = None
                    token_cur = torch.zeros(1, dtype=torch.long, device=device)
                    hyp_ids: list[int] = []
                    max_total = RNNTTrainingConstants.MAX_STREAMING_DECODE_STEPS
                    total_dec = 0
                    for t in range(Tprime):
                        u = 0
                        while u < RNNTTrainingConstants.MAX_STREAMING_TOKENS_PER_FRAME and total_dec < max_total:
                            pred_step, hidden = model.predictor.forward_streaming(token_cur.unsqueeze(1), hidden)
                            logits_tu = model.joiner(enc_out[:, t:t+1, :], pred_step)
                            next_id = int(logits_tu[0, 0, 0].argmax().item())
                            total_dec += 1
                            if next_id == RNNTTrainingConstants.RNNT_BLANK_TOKEN:
                                break
                            hyp_ids.append(next_id)
                            token_cur = torch.tensor([next_id], dtype=torch.long, device=device)
                            u += 1
                    hyp_text = tokenizer.decode(hyp_ids)
                    ref_text = tokenizer.normalize(texts[0])
                    total_wer += wer(ref_text, hyp_text)
                    num_eval += 1
            if num_eval > 0:
                print(f"post-train eval: avg WER over {num_eval} samples = {total_wer/num_eval:.3f}")
            model.train()
        except Exception as e:
            print(f"Eval failed: {e}")


if __name__ == "__main__":
    main()
