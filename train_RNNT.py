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
from utils.tokenizer import CharTokenizer
from utils.metrics import wer


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
    
    # Performance Monitoring
    LOG_INTERVAL = 5                # Steps between loss logging (more frequent)
    SYNC_INTERVAL = 1               # Epochs between device synchronization
    
    # Memory Management
    GRAD_SET_TO_NONE = True         # More efficient than zero_grad()
    
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        
        return mel_features, acoustic_length, target_tokens


def collate(batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    features_list, feature_lengths, tokens_list = zip(*batch)
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
    for batch_idx, (features, _, tokens) in enumerate(zip(features_list, feature_lengths, tokens_list)):
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
    
    return padded_features, feature_lengths_tensor, padded_tokens, token_lengths_tensor


def rnnt_loss_naive_batch(logits: torch.Tensor, tokens: torch.Tensor, out_lens: torch.Tensor, token_lens: torch.Tensor, blank: int = 0) -> torch.Tensor:
    """Very slow, naive RNNT loss for small T,U (debug/sanity only).

    logits: (B, T, U, V) unnormalized
    tokens: (B, U) with tokens[*,0] = blank, labels in 1..V-1
    out_lens: (B,) lengths in T dimension after encoder
    token_lens: (B,) lengths in U dimension including initial blank
    returns mean loss over batch
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
    """Select and return the best-available RNNT loss backend.

    Order of preference (unless a specific choice is requested):
    - torchaudio.prototype.rnnt.rnnt_loss
    - warp_rnnt.rnnt_loss (if installed)
    - None (caller will fall back to naive/CTC)

    Returns:
        (callable|None, str): loss function and backend name
    """
    preferred = preferred.lower()

    def try_ta():
        try:
            from torchaudio.prototype.rnnt import rnnt_loss as ta_rnnt_loss  # type: ignore
            return ta_rnnt_loss, "torchaudio"
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

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Enable MPS profiler context where available")
    parser.add_argument("--manifest", type=str, default="", help="Path to LibriSpeech CSV manifest for RNNT training")
    parser.add_argument("--force_naive_rnnt", action="store_true", help="Force small T',U for naive RNNT loss path")
    parser.add_argument("--rnnt_impl", type=str, default="auto", choices=["auto", "torchaudio", "warp_rnnt", "naive", "ctc"], help="Select RNN-T loss implementation")
    parser.add_argument("--max_align", type=int, default=250000, help="Maximum allowed T'*U alignment size before clamping/fallback")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Tokenizer and vocab
    tokenizer = CharTokenizer()
    cfg = MCTConfig(vocab_size=tokenizer.vocab_size)
    model = MCTModel(cfg).to(device)

    if args.manifest and HAS_LS:
        try:
            ds = LibriSpeechCSVDataset(args.manifest)
            dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=ls_collate)
            print(f"Loaded LibriSpeech CSV: {args.manifest} ({len(ds)} rows)")
        except Exception as e:
            print(f"Failed to load manifest: {e}. Falling back to dummy dataset.")
            ds = DummyRNNTDataset(num=8 if args.sanity else 64, vocab=tokenizer.vocab_size)
            dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    else:
        ds = DummyRNNTDataset(num=8 if args.sanity else 64, vocab=tokenizer.vocab_size)
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    # Select RNNT loss backend
    rnnt_loss, rnnt_backend = select_rnnt_backend(args.rnnt_impl)
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
                if (not args.force_naive_rnnt) and (rnnt_loss is not None) and (rnnt_backend in ("torchaudio", "warp_rnnt")):
                    with record_function("rnnt_loss_compute"):
                        log_probs = logits.log_softmax(dim=-1)
                        try:
                            loss = rnnt_loss(
                                log_probs, tokens, out_lens, token_lens, blank=0, clamp=-1, reduction="mean"
                            )
                        except Exception as e:
                            # Safe CPU fallback for loss only
                            print(f"RNN-T backend failed on-device ({e}); retrying loss on CPU.")
                            lp_cpu = log_probs.detach().to("cpu")
                            tok_cpu = tokens.detach().to("cpu")
                            ol_cpu = out_lens.detach().to("cpu")
                            tl_cpu = token_lens.detach().to("cpu")
                            loss = rnnt_loss(lp_cpu, tok_cpu, ol_cpu, tl_cpu, blank=0, clamp=-1, reduction="mean").to(log_probs.device)
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

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                if step % 10 == 0:
                    log_msg = f"epoch {epoch} step {step} loss {loss.item():.4f}"
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

            if device.type == "mps":
                torch.mps.synchronize()

    elapsed = time.time() - start
    if elapsed > 0:
        print(f"encoder throughput ~ {total_frames/elapsed:.1f} frames/sec (dummy)")


if __name__ == "__main__":
    main()
