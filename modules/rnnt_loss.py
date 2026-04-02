"""
Shared RNN-T loss functions for Apple Silicon MambaASR training.

This module provides reusable RNN-T loss computation functions extracted from
train_RNNT.py so they can be shared across multiple training scripts and
the MPS-native loss facade (rnnt_loss_mps.py).

Exported Functions:
- rnnt_loss_naive_batch: Reference DP-based RNN-T loss (debugging / small sequences)
- select_rnnt_backend: Backend selection with graceful fallback
- _rnnt_loss_torchaudio_safe: Per-sample torchaudio wrapper avoiding batch mismatches
- _rnnt_loss_cpu_with_grad: CPU fallback with explicit gradient computation

Cross-File Integration:
- Called by: train_RNNT.py training loop, modules/rnnt_loss_mps.py facade
- Calls: torchaudio.prototype.rnnt, torchaudio.functional, warp_rnnt backends
- Shared by: Any training script requiring RNN-T loss with Apple Silicon support

Apple Silicon Considerations:
- All functions handle MPS device tensors with CPU fallback where needed
- Per-sample processing avoids MPS operation gaps in batch RNN-T kernels
- Gradient injection pattern supports MPS autograd limitations
"""
from __future__ import annotations

import torch


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
        tgt_b = tokens_with_blank[b, 1 : 1 + Ui_eff].to("cpu")
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
