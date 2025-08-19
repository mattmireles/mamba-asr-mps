#!/usr/bin/env python3
"""
End-to-end training script for Mamba-ASR with a learned 1024→29 projection head.

This script trains a production-ready projection head on top of a 1024-class
ConMamba CTC backbone and automates validation, checkpointing, and post-run
export/evaluation.

Key design:
- Backbone: ConMambaCTC with cfg.vocab_size=1024 to produce per-frame 1024 logits
- Head: Final torch.nn.Linear(1024, 29) named `proj` to produce 29-char logits
- Loss: CTC loss over 29-character vocabulary (blank at index 0)
- Validation: CTC loss + Character Error Rate (CER) via greedy CTC decoding
- Checkpointing: Save regular checkpoints and the best checkpoint by lowest CER
- Post-run: Extract learned projection to CSV, run Core ML batch eval harness

Why this design:
- The Core ML runtime and Swift runner already operate with a 1024-wide
  vocabulary internally. By learning a 1024→29 projection on real data, we can
  restrict to the 29-character alphabet with accurate, data-driven weights.
- The learned projection is exported to `exports/projection_1024x29.csv` and
  used by `scripts/eval_batch.sh` with the existing Core ML model.

Apple Silicon specifics:
- MPS acceleration is used if available, with CPU fallback enabled for CTC
- Unified memory and careful logging to avoid excessive synchronization

Usage examples:
    # Minimal sanity check on a tiny CSV
    python Mamba-ASR-MPS/train.py \
        --train-csv /path/to/train.csv \
        --val-csv /path/to/val.csv \
        --epochs 2 --batch-size 2

    # Full training
    python Mamba-ASR-MPS/train.py \
        --train-csv data/train.csv --val-csv data/val.csv \
        --epochs 30 --batch-size 8 --lr 3e-4 --d-model 256 --n-blocks 6

    # Freeze backbone and train only the projection head (faster)
    python Mamba-ASR-MPS/train.py --freeze-backbone --epochs 10

Outputs:
- Checkpoints under Mamba-ASR-MPS/exports/checkpoints
- Best checkpoint: best.pt (lowest validation CER)
- Projection CSV: Mamba-ASR-MPS/exports/projection_1024x29.csv
- Batch eval report: Mamba-ASR-MPS/exports/CoreMLTraces/wer_cer_overview_opt.md
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


# -----------------------------
# Utility: device selection (MPS → CUDA → CPU)
# -----------------------------
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------
# Data collation for CTC
# -----------------------------
def ctc_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]):
    """Collate LibriSpeechCSVDataset samples for CTC training.

    Input items: (mel_db[T,80], feat_len, tokens[U], text)
    Returns:
      feats: (B, T_max, 80)
      feat_lens: (B,)
      targets: (sum_U,)
      target_lens: (B,)
      texts: List[str]
    """
    feats_list, feat_lens, tokens_list, texts = zip(*batch)
    B = len(batch)

    max_T = max(f.shape[0] for f in feats_list)
    feats = torch.zeros(B, max_T, DS.N_MELS, dtype=torch.float32)
    for i, f in enumerate(feats_list):
        feats[i, : f.shape[0]] = f
    feat_lens_tensor = torch.stack(list(feat_lens)).to(torch.long)

    # Concatenate targets and build per-utterance target lengths
    targets = torch.cat(list(tokens_list)).to(torch.long)
    target_lens = torch.tensor([t.shape[0] for t in tokens_list], dtype=torch.long)

    return feats, feat_lens_tensor, targets, target_lens, list(texts)


# -----------------------------
# Model wrapper with learned 1024→29 head
# -----------------------------
class MambaASRForCTC(nn.Module):
    """ConMamba backbone with learned projection head for 29-character CTC.

    - Backbone: ConMambaCTC configured to output 1024 logits per frame
    - Head: nn.Linear(1024, 29) named `proj` for extraction to CSV

    Forward returns:
      logits_29: (B, T', 29)
      out_lens: (B,) lengths after subsampling (typically T/4)
    """

    def __init__(self, d_model: int = 256, n_blocks: int = 6):
        super().__init__()
        # Backbone outputs 1024-wide logits per frame
        backbone_cfg = ConMambaCTCConfig(d_model=d_model, n_blocks=n_blocks, vocab_size=1024)
        self.backbone = ConMambaCTC(backbone_cfg)
        # Final classification head for 29-char vocabulary
        self.proj = nn.Linear(1024, 29, bias=True)

    def forward(self, feats: torch.Tensor, feat_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_1024, out_lens = self.backbone(feats, feat_lens)  # (B, T', 1024)
        logits_29 = self.proj(logits_1024)  # (B, T', 29)
        return logits_29, out_lens


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
            self.total_chars += max(1, len(r))
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


def ctc_greedy_decode(logits_29: torch.Tensor, blank_id: int = 0) -> List[List[int]]:
    """Greedy CTC decode on per-frame logits.

    Args:
      logits_29: (B, T', 29)
    Returns:
      List of token id sequences (without blanks, collapsed repeats)
    """
    with torch.no_grad():
        pred = logits_29.argmax(dim=-1)  # (B, T')
    sequences: List[List[int]] = []
    for row in pred:
        prev = blank_id
        seq: List[int] = []
        for token in row.tolist():
            if token != prev and token != blank_id:
                seq.append(token)
            prev = token
        sequences.append(seq)
    return sequences


def ids_to_text(token_ids: List[int], tokenizer: CharTokenizer) -> str:
    # Tokenizer maps 1: space, 2-27: a-z, 28: apostrophe; 0 is blank
    # Keep only known tokens
    chars: List[str] = []
    for tid in token_ids:
        if tid == 0:
            continue
        ch = tokenizer.id_to_char.get(tid)
        if ch is not None:
            chars.append(ch)
    return "".join(chars)


# -----------------------------
# Training / validation
# -----------------------------
@dataclass
class TrainConfig:
    train_csv: str
    val_csv: str
    epochs: int = 10
    batch_size: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-2
    d_model: int = 256
    n_blocks: int = 6
    num_workers: int = 0  # Apple Silicon typically prefers single-threaded IO
    checkpoint_dir: str = "Mamba-ASR-MPS/exports/checkpoints"
    eval_every_epochs: int = 1
    log_interval: int = 25
    grad_clip: float = 5.0
    freeze_backbone: bool = False
    seed: int = 42


def set_seed(seed: int) -> None:
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)


class PerformanceMonitor:
    """Lightweight monitor to estimate data wait vs. training compute time.

    Logs the proportion of time spent waiting on DataLoader vs. doing
    the train step. Intended to help tune num_workers/batch_size.
    """

    def __init__(self, log_every: int = 100):
        self.log_every = log_every
        self._last_time = time.perf_counter()
        self._data_wait_sum = 0.0
        self._train_sum = 0.0
        self._phase = "idle"

    def batch_fetch_started(self) -> None:
        now = time.perf_counter()
        if self._phase == "train":
            self._train_sum += now - self._last_time
        self._last_time = now
        self._phase = "data"

    def train_step_started(self) -> None:
        now = time.perf_counter()
        if self._phase == "data":
            self._data_wait_sum += now - self._last_time
        self._last_time = now
        self._phase = "train"

    def maybe_log(self, step: int) -> None:
        if step % self.log_every != 0:
            return
        total = self._data_wait_sum + self._train_sum
        if total > 0:
            data_pct = (self._data_wait_sum / total) * 100.0
            gpu_pct = (self._train_sum / total) * 100.0
            print(f"  [Perf] GPU-busy: {gpu_pct:.1f}% | Data-wait: {data_pct:.1f}% (over last {self.log_every} steps)")
        # reset window
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
            pred_ids_batch = ctc_greedy_decode(logits_29)
            for pred_ids, ref_text in zip(pred_ids_batch, texts):
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-blocks", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=-1, help="DataLoader workers (-1=auto-detect based on CPU cores)")
    parser.add_argument("--checkpoint-dir", type=str, default="Mamba-ASR-MPS/exports/checkpoints")
    parser.add_argument("--eval-every-epochs", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--grad-clip", type=float, default=5.0)
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

    # Auto-detect workers if requested
    worker_count = cfg.num_workers
    if worker_count == -1:
        try:
            from utils.hardware import get_optimal_worker_count
        except Exception:
            # fallback: conservative 0 (main process)
            worker_count = 0
        else:
            worker_count = get_optimal_worker_count()
        print(f"Auto-detected {worker_count} dataloader workers.")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=worker_count, collate_fn=ctc_collate, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=worker_count, collate_fn=ctc_collate, pin_memory=False)

    # Model
    model = MambaASRForCTC(d_model=cfg.d_model, n_blocks=cfg.n_blocks)
    if cfg.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        print("Backbone frozen; training projection head only")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))

    ckpt_dir = Path(cfg.checkpoint_dir)
    last_ckpt = ckpt_dir / "last.pt"
    best_ckpt = ckpt_dir / "best.pt"

    best_val_cer: float | None = None

    # Training loop
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        epoch_start = time.time()
        perf = PerformanceMonitor(log_every=max(50, cfg.log_interval))

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

            optimizer.zero_grad(set_to_none=True)
            # Guard against NaNs/Infs
            if not torch.isfinite(loss):
                continue
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
                rc = os.spawnvp(os.P_WAIT, cmd[0], cmd)
                if rc != 0:
                    print("Projection extraction failed (non-zero exit).")
                else:
                    print("Projection CSV written:", proj_out)

                # Run batch eval harness (uses learned projection automatically)
                eval_script = repo_root / "scripts/eval_batch.sh"
                if eval_script.exists():
                    print("Running batch evaluation harness...")
                    rc2 = os.spawnlp(os.P_WAIT, "bash", "bash", str(eval_script))
                    if rc2 != 0:
                        print("Batch evaluation failed (non-zero exit). See logs above.")
                    else:
                        print("Batch evaluation complete. See exports/CoreMLTraces for reports.")
                else:
                    print("Batch eval script not found; skipping.")
        except Exception as e:
            print(f"Post-run steps encountered an error: {e}")


if __name__ == "__main__":
    main()
