### RNNT baseline on Apple Silicon — operational notes (ongoing)

These notes capture concrete issues, fixes, and heuristics discovered while training the MCT (Mamba-CNN Transducer) with RNN-T loss on macOS (MPS). The goal is to turn surprises into reproducible playbooks.

#### Environment
- Python: ARM64; MPS available (verified).
- `PYTORCH_ENABLE_MPS_FALLBACK=1` enabled for initial migrations.
- Dataset: LibriSpeech dev-clean CSV manifest (512 rows slice for quick runs).

#### Observations and fixes
- Torchaudio RNN-T API surface
  - `torchaudio.prototype.rnnt.rnnt_loss` preferred; `torchaudio.functional.rnnt_loss` is deprecated (removed in 2.9).
  - Torchaudio often throws “input/output length mismatch” on batched inputs unless sequences are sliced per-sample to exact `(Ti, Ui)` caps and targets exclude the leading blank. We implemented per-sample slicing in both the fast path and CPU-grad fallback.

- Non-finite loss (NaN) emergence
  - Finite and stable for ~0–200 steps, then sporadic NaNs begin to appear when alignment grids get large.
  - Hypothesis: numerical instability rises with large T'·U. We observed T'·U in the 90k–138k range preceding NaNs.
  - Mitigation implemented:
    - `--max_align` cap plus per-batch U-capping in the torchaudio path to keep T'·U below threshold.
    - `--grad_clip` (global norm) and `--skip_non_finite` to continue training while skipping bad steps.

- CPU-grad RNNT fallback
  - We implemented `_rnnt_loss_cpu_with_grad`: copy logits to CPU with `requires_grad_(True)`, compute per-sample RNNT loss, `backward()` on CPU to get logits.grad, then backprop into the MPS graph via `logits.backward(grad_logits)`.
  - This path is robust when torchaudio rejects batched shapes; it keeps training moving at the expense of throughput.

#### Concrete runs (latest)
- Extended run (torchaudio backend; bs=2; 512 samples; 400 steps)
  - Loss: 4.46 → ~2.86 by ~step 190, then NaNs after ~200
  - Throughput: ~1584 fps (encoder)
  - WER: ~1.000 (early; expected)

- Guarded run (max_align=80k; grad_clip=1.0; skip_non_finite)
  - Non-finite steps appear after ~120, are skipped; training continues
  - Throughput: ~1878.6 fps

- CPU-grad baseline (max_align=60k; no forced cpu flag; torchaudio frequently falls back)
  - Loss snapshots (CPU-grad logs): 316.8 → 140.0 → 325.0 → 307.4 → 262.6 → 263.8 → 418.8 → 319.8 → 196.1 → 456.7 → 257.4 → 115.1 → 99.4 → 182.8 → 149.2 → 91.9 → 116.9 → 281.8 → 240.6 → 504.4 → 196.8 → 55.9 → 368.0 → 310.9 → 234.3 → 227.8
  - Throughput: ~1248.0 fps (encoder)
  - WER: ~1.000 (early; expected)

- CPU-grad baseline (dev-clean slice; 120 steps; forced cpu-grad)
  - Command summary: bs=2, max_samples=256, max_steps=120, `--force_cpu_grad`, `--max_align 60000`, `--grad_clip 1.0`, `--skip_non_finite`
  - Loss snapshots: 440.53 → 75.25 → 102.05 → 51.28 (every 10 steps)
  - Throughput: encoder ~1766.4 fps (bs=2)
  - Align stats: p50=3,918; p90=5,261; p99=5,677; max=5,694
  - T' caps: p50=132; p90=146; max=149. U caps: p50=34; p90=39; max=40
  - Backend usage: 100% `cpu_grad`
  - Artifacts: checkpoint `checkpoints/rnnt_devclean_cpu_grad_120.pt`, csv `logs/rnnt_devclean_cpu_grad_120.csv`

- Auto-backend run (dev-clean slice; 120 steps; guards on)
  - Behavior: torchaudio selected; per-batch CPU-grad fallback due to length mismatches
  - Throughput: encoder ~1709.6 fps (bs=2)
  - Loss snapshots: 481.87 → 92.60 → 97.74 → 33.69 (every 10 steps)
  - Align stats: p50=3,266; p90=4,954; p99=5,542; max=5,655
  - T' caps: p50=132; p90=147; max=148. U caps: p50=24; p90=36; max=40
  - Backend usage: 100% `cpu_grad`
  - Artifacts: checkpoint `checkpoints/rnnt_devclean_auto_120.pt`, csv `logs/rnnt_devclean_auto_120.csv`

#### Heuristics that worked
- Always compute/slice per-sample `(Ti, Ui)` before RNNT loss; exclude leading blank from targets.
- Cap T'·U via `--max_align` and shrink U across the batch when necessary.
- Enable `--skip_non_finite` and `--grad_clip` for long runs; do not crash the epoch.
- Keep a CPU-grad RNNT path available; it is essential to maintain progress when the fast path rejects shapes.

#### Instrumentation
- Log `align(T'U')` per batch along with loss.
- Periodic greedy RNN-T decode for a rough WER signal using streaming predictor and joiner (fast, approximate).

#### Next actions
- Tighten alignment cap further (e.g., 50–60k) and make per-sample U-capping the default in fast path.
- Attempt `warp_rnnt` install with `--no-build-isolation`; evaluate stability vs. torchaudio on Apple Silicon.
- Run a longer CPU-grad baseline epoch to capture a clean loss trajectory (stability over speed) to finalize Phase 2 baseline.
- Begin collecting stable checkpoints for Phase 3 KD/QAT/pruning experiments.
  - Update: Added end-of-run checkpoint writer in `train_RNNT.py` via `--save_ckpt` (defaults to `Mamba-ASR-MPS/checkpoints/rnnt_<ts>.pt`).
  - Update: Deprecated `--rnnt_cpu_grad`; introduced `--force_cpu_grad` (forces CPU-grad every batch). Automatic CPU-grad fallback still occurs when fast backend fails.
  - Update: Phase 3 artifacts now saved via `scripts/optimize.py --save_model` for KD/QAT/Prune.
  - KD short pass (dev-clean slice): avg_loss≈2.8510; encoder throughput≈2563.2 fps
  - QAT short pass: last_loss≈0.0; encoder throughput≈2942.4 fps (fake-quant; deprecation suggests torchao PT2E migration)
  - Structured pruning short pass: last_loss≈0.0; encoder throughput≈2474.1 fps

#### Core ML streaming latency (Swift runner)
- Command:
  ```bash
  python Mamba-ASR-MPS/scripts/export_and_validate.py \
    --checkpoint Mamba-ASR-MPS/checkpoints/kd_student_auto.pt \
    --name MambaASR_kd_auto --duration 5
  ```
- Results (256-frame chunks):
  - Before warmup: chunk0≈146.5 ms; steady ≈13.4/13.2 ms; avg≈57.7 ms
  - With `--warmup 2` (new flag in runner): KD avg≈14.7 ms; QAT avg≈13.3 ms; Pruned avg≈13.5 ms (3 chunks)
  - CSV capture (10s runs):
    - KD: `--latency-csv Mamba-ASR-MPS/exports/latency_kd_auto.csv` (avg≈17.1 ms, n=4)
    - QAT: `--latency-csv Mamba-ASR-MPS/exports/latency_qat_auto.csv` (avg≈16.2 ms, n=4)
    - Pruned: `--latency-csv Mamba-ASR-MPS/exports/latency_pruned_auto.csv` (avg≈18.2 ms, n=4)
  - Steady-state per-chunk ≈13–15 ms → ~0.05–0.06 s model-only time for a 10 s clip (4 chunks) post-warmup
  - Action: verify ANE execution via Activity Monitor (Neural Engine) during streaming; test with `--wav` real audio

### Swift streaming (20s) for ANE verification (today)
- QAT (20s): avg≈13.01 ms; p50≈12.95; p90≈13.31; n=8 → `exports/latency_MambaASR_qat_real_20s.csv`
- Pruned (20s): avg≈13.25 ms; p50≈13.29; p90≈13.88; n=8 → `exports/latency_MambaASR_pruned_real_20s.csv`
- Note: Watch Activity Monitor → GPU History → Neural Engine for utilization during runs.

### Per-layer structured pruning (today)
- Ran `scripts/optimize.py --technique prune --sparsity_map '{"Conv1d":0.30,"Conv2d":0.40,"Linear":0.30}' --save_model checkpoints/pruned_layered.pt`
- Exported and validated: avg≈12.92 ms; p50≈12.95; p90≈13.20; n=8 → `exports/latency_MambaASR_pruned_layered.csv`
- `scripts/optimize.py` now supports `--sparsity` and `--sparsity_map` for per-layer targets.
### RNNT CPU-grad run (dev-clean; 150 steps; today)
- Cmd:
  ```bash
  PYTORCH_ENABLE_MPS_FALLBACK=1 \
  PYTHONPATH="$(pwd)/Mamba-ASR-MPS" \
  python Mamba-ASR-MPS/train_RNNT.py \
    --epochs 1 --batch_size 2 \
    --manifest "/Users/mattmireles/Documents/Training Data/LibriSpeech/dev-clean.csv" \
    --num_workers 0 --max_samples 512 --max_steps 150 \
    --device mps --rnnt_impl auto --force_cpu_grad \
    --max_align 60000 --grad_clip 1.0 --skip_non_finite \
    --log_csv Mamba-ASR-MPS/logs/rnnt_devclean_cpu_grad_150.csv \
    --save_ckpt Mamba-ASR-MPS/checkpoints/rnnt_devclean_cpu_grad_150.pt \
    --eval_after --eval_samples 64
  ```
- Observations:
  - Backend: 100% cpu_grad engaged; torchaudio RNNT unsupported on MPS (expected)
  - Throughput: encoder ~1405.8 fps (bs=2)
  - Loss snapshots: 481.84 → 49.63 → 81.72 → 55.69 (every ~10 steps)
  - Align stats: count=32; p50=4012; p90=4759; p99=5346; max=5421
  - T' caps: p50=124; p90=143; max=148; U caps: p50=32; p90=40; max=40
  - Greedy WER (quick eval): ~1.000 (early-stage, expected)
  - Artifacts: `logs/rnnt_devclean_cpu_grad_150.csv`, `checkpoints/rnnt_devclean_cpu_grad_150.pt`

### RNNT CPU-grad run (dev-clean; 300 steps; today)
- Cmd:
  ```bash
  PYTORCH_ENABLE_MPS_FALLBACK=1 \
  PYTHONPATH="$(pwd)/Mamba-ASR-MPS" \
  python Mamba-ASR-MPS/train_RNNT.py \
    --epochs 1 --batch_size 2 \
    --manifest "/Users/mattmireles/Documents/Training Data/LibriSpeech/dev-clean.csv" \
    --num_workers 0 --max_samples 512 --max_steps 300 \
    --device mps --rnnt_impl auto --force_cpu_grad \
    --max_align 60000 --grad_clip 1.0 --skip_non_finite \
    --log_csv Mamba-ASR-MPS/logs/rnnt_devclean_cpu_grad_300.csv \
    --save_ckpt Mamba-ASR-MPS/checkpoints/rnnt_devclean_cpu_grad_300.pt \
    --eval_after --eval_samples 96
  ```
- Observations:
  - Throughput: encoder ~1468.4 fps (bs=2)
  - Loss snapshots: 404.70 → 113.87 → 91.03 → 66.74 (every ~10 steps)
  - Align stats: count=32; p50=3610; p90=4960; p99=5273; max=5282
  - T' caps: p50=122; p90=139; max=149; U caps: p50=32; p90=38; max=40
  - Backend usage: 100% cpu_grad
  - Artifacts: `logs/rnnt_devclean_cpu_grad_300.csv`, `checkpoints/rnnt_devclean_cpu_grad_300.pt`

### RNNT CPU-grad run (dev-clean; 600 steps; today)
- Cmd:
  ```bash
  PYTORCH_ENABLE_MPS_FALLBACK=1 \
  PYTHONPATH="$(pwd)/Mamba-ASR-MPS" \
  python Mamba-ASR-MPS/train_RNNT.py \
    --epochs 1 --batch_size 2 \
    --manifest "/Users/mattmireles/Documents/Training Data/LibriSpeech/dev-clean.csv" \
    --num_workers 0 --max_samples 512 --max_steps 600 \
    --device mps --rnnt_impl auto --force_cpu_grad \
    --max_align 60000 --grad_clip 1.0 --skip_non_finite \
    --log_csv Mamba-ASR-MPS/logs/rnnt_devclean_cpu_grad_600.csv \
    --save_ckpt Mamba-ASR-MPS/checkpoints/rnnt_devclean_cpu_grad_600.pt \
    --eval_after --eval_samples 128
  ```
- Observations:
  - Throughput: encoder ~1987.7 fps (bs=2)
  - Loss snapshots: 412.91 → 36.51 → 123.57 → 89.34 (every ~10 steps)
  - Align stats: count=32; p50=3668; p90=4860; p99=5120; max=5160
  - T' caps: p50=124; p90=141; max=147; U caps: p50=33; p90=40; max=40
  - Backend usage: 100% cpu_grad
  - Artifacts: `logs/rnnt_devclean_cpu_grad_600.csv`, `checkpoints/rnnt_devclean_cpu_grad_600.pt`

### RNNT CPU-grad run (dev-clean; 800 steps; today)
- Cmd:
  ```bash
  PYTORCH_ENABLE_MPS_FALLBACK=1 \
  PYTHONPATH="$(pwd)/Mamba-ASR-MPS" \
  python Mamba-ASR-MPS/train_RNNT.py \
    --epochs 1 --batch_size 2 \
    --manifest "/Users/mattmireles/Documents/Training Data/LibriSpeech/dev-clean.csv" \
    --num_workers 0 --max_samples 512 --max_steps 800 \
    --device mps --rnnt_impl auto --force_cpu_grad \
    --max_align 60000 --grad_clip 1.0 --skip_non_finite \
    --log_csv Mamba-ASR-MPS/logs/rnnt_devclean_cpu_grad_800.csv \
    --save_ckpt Mamba-ASR-MPS/checkpoints/rnnt_devclean_cpu_grad_800.pt \
    --eval_after --eval_samples 160
  ```
- Observations:
  - Throughput: encoder ~2006.9 fps (bs=2)
  - Loss snapshots: 455.68 → 106.93 → 132.53 → 113.31 (every ~10 steps)
  - Align stats: count=32; p50=3794; p90=5662; p99=5729; max=5733
  - T' caps: p50=128; p90=149; max=149; U caps: p50=33; p90=40; max=40
  - Backend usage: 100% cpu_grad
  - Artifacts: `logs/rnnt_devclean_cpu_grad_800.csv`, `checkpoints/rnnt_devclean_cpu_grad_800.pt`

#### Metrics to track going forward
- Fraction of batches using CPU-grad fallback vs. fast path.
- Non-finite skip rate per N steps.
- T' and U histograms and T'·U distribution over the corpus.
- Encoder throughput and wall-clock per step.

Recent CSV stats (short CPU-grad-heavy run):
- align p50=34,599; p90=78,967; p99≈128,078; max=130,240
- T' caps p50=240; p90=364; max=484
- U caps p50=150; p90=223; max=296
- backend usage: 100% `cpu_grad` for this run (torchaudio rejected batches frequently)

Quick A/B (128 samples, 120 steps; max_align=60k):
- Without `--rnnt_cpu_grad` (auto fallback):
  - align p50≈3,483; p90≈4,702; max≈5,472; T' p50≈128 (max 147); U p50≈28 (max 40)
  - backend: 100% `cpu_grad` (torchaudio rejected batches consistently)
- With `--rnnt_cpu_grad`:
  - align p50≈3,830; p90≈5,133; max≈5,513; T' p50≈126 (max 149); U p50≈32 (max 40)
  - backend: 100% `cpu_grad` (explicit)
Interpretation: For these settings, both modes effectively exercised the CPU-grad path; explicit flag gave slightly larger U, similar T', consistent stability. We will attempt a maintained RNNT op to reduce CPU usage.

Tighter alignment cap test (50k; 256 samples, 200 steps; CPU-grad):
- align p50≈3,450; p90≈4,703; p99≈5,578; max≈5,920
- T' p50≈121; p90≈142; max≈148
- U p50≈28; p90≈37; max≈40
- Throughput: ~1,834.5 fps (encoder)
- Backend: 100% `cpu_grad` for this run

#### Gotchas (Apple Silicon / MPS)
- CTC used in auxiliary paths auto-falls back to CPU; this is fine for our usage but shows CPU logs.
- `.item()` in tight loops forces sync; only compute aggregated stats when needed.

#### One-liners
- Current stable run template (guarded):
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
PYTHONPATH="$(pwd)/Mamba-ASR-MPS" \
python Mamba-ASR-MPS/train_RNNT.py \
  --epochs 1 --batch_size 2 \
  --manifest "/Users/mattmireles/Documents/Training Data/LibriSpeech/dev-clean.csv" \
  --num_workers 0 --max_samples 512 --max_steps 300 \
  --device mps --rnnt_impl auto \
  --max_align 60000 --grad_clip 1.0 --skip_non_finite \
  --eval_after --eval_samples 64
```

- Core ML validation (Swift runner):
```bash
xcrun coremlcompiler compile \
  Mamba-ASR-MPS/exports/MambaASR.mlpackage \
  Mamba-ASR-MPS/exports/Compiled

swift/MambaASRRunner/.build/release/MambaASRRunner \
  --mlmodelc Mamba-ASR-MPS/exports/Compiled/MambaASR.mlmodelc \
  --mlpackage Mamba-ASR-MPS/exports/MambaASR.mlpackage \
  --stream  # optional; add --wav /path/to/16k_mono.wav to stream real audio
```
Result: success; shapes `logits_time=[1,64,1,1024]`, `predictor_hidden_out=[1,1,256]`. Streaming heartbeat: last-step argmax per chunk.

- Phase 3 short passes with saving:
```bash
python Mamba-ASR-MPS/scripts/optimize.py --technique kd \
  --manifest "/Users/mattmireles/Documents/Training Data/LibriSpeech/dev-clean.csv" \
  --steps 50 --batch_size 2 \
  --save_model Mamba-ASR-MPS/checkpoints/kd_student.pt

python Mamba-ASR-MPS/scripts/optimize.py --technique qat \
  --manifest "/Users/mattmireles/Documents/Training Data/LibriSpeech/dev-clean.csv" \
  --steps 50 --batch_size 2 \
  --save_model Mamba-ASR-MPS/checkpoints/qat_model.pt

python Mamba-ASR-MPS/scripts/optimize.py --technique prune \
  --manifest "/Users/mattmireles/Documents/Training Data/LibriSpeech/dev-clean.csv" \
  --steps 50 --batch_size 2 \
  --save_model Mamba-ASR-MPS/checkpoints/pruned_model.pt
```
