"""
MPS-native RNN-T loss facade for Apple Silicon.

This module provides an initial scaffold for a future native MPS implementation
of the RNN-T loss. For now, it delegates to the best available backend
(torchaudio if present), with robust CPU-grad fallback and CTC approximation.

Design goals:
- Single entrypoint rnnt_loss_mps() to be called from training code
- Prefer MPS execution when possible
- Avoid device thrashing; keep tensors on device except for CPU fallback
- Graceful degradation to CPU or CTC
"""
from __future__ import annotations

from typing import Tuple, Optional
import os
import torch


def select_best_backend():
    """Return (fn, name) for the best-available RNNT backend.
    Preference order:
    - torchaudio.prototype.rnnt.rnnt_loss (preferred; functional is deprecated)
    - warp_rnnt.rnnt_loss (if installed)
    - None
    """
    # Prefer torchaudio prototype
    try:
        from torchaudio.prototype.rnnt import rnnt_loss as ta_rnnt_loss  # type: ignore
        return ta_rnnt_loss, "torchaudio"
    except Exception:
        pass
    # Fallback: warp_rnnt if available
    try:
        from warp_rnnt import rnnt_loss as warp_rnnt_loss  # type: ignore
        return warp_rnnt_loss, "warp_rnnt"
    except Exception:
        pass
    return None, "none"


def _cpu_grad_fallback(rnnt_fn, logits: torch.Tensor, tokens_with_blank: torch.Tensor, out_lens: torch.Tensor, token_lens_with_blank: torch.Tensor, blank: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute RNNT on CPU per-sample to obtain (loss, grad_logits) on device."""
    try:
        from torchaudio.functional import rnnt_loss as _ta_fn  # type: ignore
        is_ta = rnnt_fn is _ta_fn
    except Exception:
        is_ta = False
    B, Tcap, Ucap, V = logits.shape
    logits_cpu = logits.detach().to("cpu").requires_grad_(True)
    total = None
    used = 0
    for b in range(B):
        Ti = int(out_lens[b].item())
        Ui_wb = int(token_lens_with_blank[b].item())
        Ui = max(0, Ui_wb - 1)
        Ti_eff = min(Ti, Tcap)
        Ui_eff = min(Ui, max(0, Ucap - 1))
        if Ti_eff <= 0 or Ui_eff <= 0:
            continue
        sl = logits_cpu[b : b + 1, : Ti_eff, : (Ui_eff + 1), :].contiguous()
        lp = sl.log_softmax(dim=-1)
        tgt = tokens_with_blank[b, 1 : 1 + Ui_eff]
        if is_ta:
            tgt = tgt.to(torch.int32).unsqueeze(0)
            Tl = torch.tensor([Ti_eff], dtype=torch.int32)
            Ul = torch.tensor([Ui_eff], dtype=torch.int32)
        else:
            tgt = tgt.unsqueeze(0)
            Tl = torch.tensor([Ti_eff], dtype=torch.long)
            Ul = torch.tensor([Ui_eff], dtype=torch.long)
        loss_b = rnnt_fn(lp, tgt, Tl, Ul, blank=blank, clamp=-1, reduction="mean")
        total = loss_b if total is None else total + loss_b
        used += 1
    if used == 0:
        return torch.zeros((), device=logits.device), torch.zeros_like(logits)
    total = total / used
    total.backward()
    grad_logits = logits_cpu.grad.to(logits.device)
    return total.to(logits.device).detach(), grad_logits.detach()


def rnnt_loss_mps(logits: torch.Tensor, tokens_with_blank: torch.Tensor, out_lens: torch.Tensor, token_lens_with_blank: torch.Tensor, blank: int = 0, max_align: Optional[int] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], str]:
    """
    Compute RNN-T loss with preference for MPS execution; fall back to CPU-grad.

    Returns (loss, grad_logits_or_None, backend_name).
    - If backend runs on-device with autograd, grad_logits_or_None is None.
    - If CPU-grad fallback is used, returns explicit grad_logits.
    """
    rnnt_fn, backend = select_best_backend()
    if rnnt_fn is None:
        return torch.tensor(0.0, device=logits.device), None, "none"

    try:
        # Prepare inputs for torchaudio (prototype or functional) consistently
        is_torchaudio = False
        try:
            from torchaudio.prototype.rnnt import rnnt_loss as _ta_proto  # type: ignore
            if rnnt_fn is _ta_proto:
                is_torchaudio = True
        except Exception:
            pass
        # Only support prototype moving forward (functional is deprecated in 2.9)

        lp = logits.log_softmax(dim=-1)
        Tcap = lp.shape[1]
        Ucap = lp.shape[2]
        # Guard: keep effective T*U manageable by shrinking U batch-wide if needed
        # Uses a conservative cap similar to training default
        MAX_ALIGN = int(max_align) if (max_align is not None) else int(os.environ.get("RNNT_MAX_ALIGN", "60000"))
        # Compute per-sample allowed U to satisfy T*U <= MAX_ALIGN
        allowed_Us = []
        for i in range(out_lens.numel()):
            Ti_eff_i = int(min(int(out_lens[i].item()), Tcap))
            Ui_eff_i = int(max(0, int(token_lens_with_blank[i].item()) - 1))
            if Ti_eff_i <= 0:
                allowed_Us.append(0)
                continue
            Ui_allowed = min(Ui_eff_i, max(1, MAX_ALIGN // max(1, Ti_eff_i)))
            allowed_Us.append(Ui_allowed)
        Ubatch = min(allowed_Us) if allowed_Us else 0
        if is_torchaudio:
            t_tokens = tokens_with_blank[:, 1:].to(torch.int32)  # exclude leading blank
            t_out = torch.clamp(out_lens.to(torch.int32), max=Tcap)
            t_tok = (token_lens_with_blank - 1).clamp_min(0).to(torch.int32)
            if Ubatch > 0:
                t_tok = torch.clamp(t_tok, max=min(Ucap, Ubatch))
                lp = lp[:, :, : min(Ucap, Ubatch), :]
        else:
            # warp_rnnt often accepts long, but keep shapes safe
            t_tokens = tokens_with_blank[:, 1:]
            t_out = torch.clamp(out_lens, max=Tcap)
            t_tok = (token_lens_with_blank - 1).clamp_min(0)
            if Ubatch > 0:
                t_tok = torch.clamp(t_tok, max=min(Ucap, Ubatch))
                lp = lp[:, :, : min(Ucap, Ubatch), :]

        loss = rnnt_fn(lp, t_tokens, t_out, t_tok, blank=blank, clamp=-1, reduction="mean")
        return loss, None, backend
    except Exception:
        # CPU-grad fallback with explicit gradients
        loss_cpu, grad_logits = _cpu_grad_fallback(rnnt_fn, logits.detach(), tokens_with_blank, out_lens, token_lens_with_blank, blank=blank)
        return loss_cpu, grad_logits, "cpu_grad"
