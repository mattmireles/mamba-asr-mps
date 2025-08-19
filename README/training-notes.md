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

- mps_native facade (today)
  - Default RNNT backend is now `mps_native` (auto-select best backend with robust CPU-grad fallback).
  - Added dynamic U-capping in the facade: slices LogProb U-dimension and clamps token lengths using `RNNT_MAX_ALIGN` (CLI `--rnnt_max_align`).
  - Added `--adaptive_ctc_after_cpu_grad N` to automatically switch remaining batches to encoder-CTC after N consecutive `cpu_grad` RNNT batches (stability over speed).
  - Added summary logging with `--log_json` to capture encoder fps, T'/U distributions, and backend mix at end of run.

- selective_scan (today)
  - Extended benchmark to 8192 tokens on MPS shows stable/high throughput with mild variance; naive scan is not a bottleneck for short/medium chunks. Metal kernel de-prioritized.
  - Scripted report generator publishes `exports/bench_selective_scan.md` for archival.

#### Decoding and evaluation (today)
- Symptom: Core ML runner produced empty/garbage transcripts; WER/CER=1.000 across models.
- Root cause: Vocabulary mismatch. Exported models output logits over V=1024, while our intended tokenizer is 29 chars (blank, space, a–z, apostrophe). A naive modulo mapping is incorrect, yielding meaningless text.
- Remediations in `MambaASRRunner`:
  - Short-audio path: pad <256-frame clips to exactly one 256-frame chunk and run a single inference to force a transcript.
  - Pooled-greedy: log-sum-exp pool 1024 logits into 29 groups per frame; optional `--blank-gate` to avoid all-blank dominance.
  - Projection options: `--project-mod29` (fast hack) and `--proj-matrix` (now implemented) to coerce 1024→29 via learned log-weights.
    - New implementation: If `--restrict-vocab 29` and `--proj-matrix P.csv` are provided, pooled logits are computed as `pooled[k] = logsumexp_i(lps[i] + P[i,k])`.
    - CSV expected as V×29 in log-space. Falls back to modulo pooling if CSV missing/invalid.
    - Sample stub written to `exports/projection_1024x29.sample.csv` as documentation; replace with real weights.
  - Result: Text-like outputs appear, but accuracy remains poor (CER ~0.86–0.89 on repeated "hello world"; WER=1.000). Confirms need for a proper 29-vocab model or a learned projection.
- Metrics script updates: `scripts/compute_wer_cer.py` now normalizes text, computes CER character-level (spaces removed), supports `--cer-only`, thresholds, and `--strict` gating for CI.

##### Batch eval (today)
- Added `scripts/eval_batch.sh` to iterate over `exports/testset/audio/*.wav`, run greedy decode with `--restrict-vocab 29 --blank-gate 0.5 --proj-matrix`, and summarize CER via `scripts/compute_wer_cer.py --cer-only`.
- Next: Populate `exports/testset/{audio,refs}` with ~10 short 16k mono WAVs plus refs; target CER < 0.6 as an initial gate until vocab-aligned models arrive.
- Next for researchers:
  - Preferred: retrain/export with `MCTConfig(vocab_size=29)` to align logits with tokenizer and decoding.
  - Interim: add a learned 1024×29 projection head (post-export) or load a weight matrix into the Swift runner and compute y = log_softmax(W^T·softmax(x)). Expect improvement if W is derived from a trained head.
  - Keep CER as primary gating metric until the vocab alignment is fixed; treat WER as informational.

#### Concrete runs (latest)
##### Quick synthetic RNNT micro-benchmarks (sanity, bs=2)
- 60-step comparison (exports/*.csv & *.summary.json):
  - mps_native: ~1413 fps (mix of `ta`/`cpu_grad`, often `cpu_grad` dominant)
  - auto: ~1209 fps (100% `cpu_grad` observed in these short runs)
  - force_cpu_grad: ~1252 fps (100% `cpu_grad` by design)
  - ctc: ~1193 fps (encoder-only approximation)
- Longer mps_native run (180 steps): encoder_fps≈1098; summary saved to `exports/rnnt_mps_native_180.summary.json`.

##### selective_scan benchmarks
- Extended table up to 8192 tokens published at `exports/bench_selective_scan.md`; naive loop acceptable for our operating sequence lengths on MPS.
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
- Use `--adaptive_ctc_after_cpu_grad` to pivot to CTC when torchaudio repeatedly forces CPU-grad, to stabilize throughput on long runs.

#### Instrumentation
- Log `align(T'U')` per batch along with loss.
- Periodic greedy RNN-T decode for a rough WER signal using streaming predictor and joiner (fast, approximate).
- `--log_json` persists summary telemetry (fps, T'/U stats, backend usage) for dashboards.
 - Runner now logs model load timings: `compile_ms`, `instantiate_ms`, `total_ms`.
 - Added CSV summarizer: `scripts/summarize_latency_csv.py` → writes `exports/CoreMLTraces/latency_summary.md` with count/avg/p50/p90/p99.

##### Core ML trace notes (today)
- CLI exports captured under `exports/CoreMLTraces/`:
  - `quick_probe_toc.xml`, `fp16_w8_analysis_toc.xml`
  - `os_signpost_coreml.xml` (schema only, no rows exported)
  - `mps_hw_intervals.xml`, `ane_hw_intervals.xml`
- Observation: `xcrun xctrace export` did not yield per-op rows for `coreml-os-signpost` on this setup. Instruments UI is required to enumerate CPU-bound ops (Core ML → Operations → sort by Location).
- Action: Open `fp16_w8_analysis.trace` in Instruments and list CPU ops to update the remediation table in `implementation-plan-v2.md`.
- Helper scripts:
  - CSV → remediation table:
    ```bash
    python Mamba-ASR-MPS/scripts/coreml_ops_remediation.py \
      --csv Mamba-ASR-MPS/exports/CoreMLTraces/operations.csv \
      --out Mamba-ASR-MPS/exports/CoreMLTraces/coreml_cpu_ops.md
    ```
  - MPS intervals summary:
    ```bash
    python Mamba-ASR-MPS/scripts/summarize_mps_intervals.py \
      --xml Mamba-ASR-MPS/exports/CoreMLTraces/mps_hw_intervals.xml \
      --out Mamba-ASR-MPS/exports/CoreMLTraces/mps_intervals_summary.md
    ```

#### Graph-level rewrites (today)
- Rationale: CPU-only Core ML path is fastest for this model. Simplify PyTorch graphs to ops that Core ML maps efficiently on CPU.
- Selective scan inner product: replaced `torch.einsum("bdn,bn->bd", hidden_state, C_timestep)` with batched matmul to improve CPU perf and export friendliness.
  - New code: `y_timestep = torch.bmm(hidden_state, C_timestep.unsqueeze(-1)).squeeze(-1)`
  - Location: `modules/mamba/selective_scan_interface.py` inside `ss_time_loop`.
  - Result: Numerically identical; removes einsum; exported successfully to `.mlpackage`.
- Gather usage: Core ML warns that embedding/gather only supports weights+indices. Audit dynamic `gather` sites next; prefer slice/concat or embedding where applicable. (Planned)
- RNNT stability: torchaudio RNNT remains CPU-only on MPS; our facade forces CPU-grad mapping when needed. This keeps training moving but confirms RNNT is tactical; encoder optimization + CTC fallback remain strategic.
- Export check: ensure `PYTHONPATH` includes `Mamba-ASR-MPS` when invoking exporter.
  ```bash
  PYTHONPATH="$(pwd)/Mamba-ASR-MPS" \
  python Mamba-ASR-MPS/scripts/export_coreml.py \
    --output Mamba-ASR-MPS/exports/MambaASR_opt.mlpackage
  ```
- Reporting: use `scripts/report_phase3.sh` to sweep compute modes/chunks and auto-append latency summaries to the plan. Focus on the `cpu` column for regression tracking.

#### CPU chunk-size sweep (today)
- Base compiled model (`Compiled_fp16_w8`), CPU-only, chunks=[128,256,512]
  - 128: avg 4.844 ms; 256: avg 4.465 ms; 512: avg 6.740 ms
  - Tail grows at 512 (p90≈10.28 ms). Keep default chunk at 256.

#### Runner/Exporter defaults (today)
- Runner (`MambaASRRunner`) default compute now configurable via env `MAMBA_COMPUTE_DEFAULT` (default `cpu`).
- Exporter (`scripts/export_coreml.py`) default chunk length configurable via env `MAMBA_CHUNK_DEFAULT` (default 256).

#### Next actions
- Tighten alignment cap further (e.g., 50–60k) and make per-sample U-capping the default in fast path.
- Attempt `warp_rnnt` install with `--no-build-isolation`; evaluate stability vs. torchaudio on Apple Silicon.
- Run a longer CPU-grad baseline epoch to capture a clean loss trajectory (stability over speed) to finalize Phase 2 baseline.
 - Fold manual timing into CI: run runner, collect `latency_probe.csv`, summarize to markdown, and attach to plan automatically.
- Begin collecting stable checkpoints for Phase 3 KD/QAT/pruning experiments.
  - Update: Added end-of-run checkpoint writer in `train_RNNT.py` via `--save_ckpt` (defaults to `Mamba-ASR-MPS/checkpoints/rnnt_<ts>.pt`).
  - Update: Deprecated `--rnnt_cpu_grad`; introduced `--force_cpu_grad` (forces CPU-grad every batch). Automatic CPU-grad fallback still occurs when fast backend fails.
  - Update: Phase 3 artifacts now saved via `scripts/optimize.py --save_model` for KD/QAT/Prune.
  - KD short pass (dev-clean slice): avg_loss≈2.8510; encoder throughput≈2563.2 fps
  - QAT short pass: last_loss≈0.0; encoder throughput≈2942.4 fps (fake-quant; deprecation suggests torchao PT2E migration)
  - Structured pruning short pass: last_loss≈0.0; encoder throughput≈2474.1 fps

- Core ML analysis (planned)
  - Artifacts captured: `exports/CoreMLTraces/quick_probe.trace`, `fp16_w8_analysis.trace` (+ TOC XMLs). CLI export of per-op CPU list is limited; enumerate CPU ops in Instruments UI (Operations view → Location=CPU) and copy into plan remediation table.

##### One-liners (bench harness)
```bash
python Mamba-ASR-MPS/scripts/bench_rnnt_impls.py
cat Mamba-ASR-MPS/exports/bench_rnnt_summary.md
```

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

##### Beam step AMX-friendly optimization (today)
- Changes:
  - `logSoftmax` now returns `[Float]` (was `[Double]`), using vDSP/vForce and logf.
  - Introduced `logSumExpF(Float,Float)`; kept Double variant for docs consistency.
  - Replaced `Dictionary<String,CTCBeamEntry>` with `Dictionary<[Int],CTCBeamEntry>` to avoid per-frame string joins.
  - Implemented iterative top-K (O(V·K)) instead of sorting full vocab each frame.
- Result (pruned_layered; 10s; warmup=2; beam=3; real WAV): avg≈16.13 ms; p50≈15.66; p90≈16.93; n=8.
- Prior beam=3 baseline on same model: avg≈18.51 ms. Net ~14% faster with tighter tail.
 - With `--topk 6`: avg≈17.25 ms; p50≈16.82; p90≈17.86; n=8. Default top-K heuristic (max(3*beam,10)) remains preferable on CPU.
 - With `--blank-gate 0.5`: avg≈16.97 ms; p50≈16.23; p90≈18.74; n=8. Gating offered no latency benefit on this input; disabled by default.

### Swift streaming (20s) for ANE verification (today)
- QAT (20s): avg≈13.01 ms; p50≈12.95; p90≈13.31; n=8 → `exports/latency_MambaASR_qat_real_20s.csv`
- Pruned (20s): avg≈13.25 ms; p50≈13.29; p90≈13.88; n=8 → `exports/latency_MambaASR_pruned_real_20s.csv`
- Note: Watch Activity Monitor → GPU History → Neural Engine for utilization during runs.

### Beam width sweep helper (today)
- Added `--beam-list 1,3,5` to sweep beams in a single run.
- Pruned_layered (10s; warmup=2; real WAV, Float32 decoder):
  - beam=1: avg≈13.58 ms; p50≈13.42; p90≈14.09; n=8
  - beam=3: avg≈13.00 ms; p50≈12.92; p90≈13.56; n=8
  - beam=5: avg≈13.08 ms; p50≈13.06; p90≈13.21; n=8
– Takeaway: AMX-friendly path reduces beam overhead; 1–5 has similar latency on CPU now.

### Per-layer structured pruning (today)
### RNNT guard tightening + profiling spans (today)
- Changes:
  - Default `--max_align` tightened to 60k in `train_RNNT.py` based on observed T'·U histograms.
  - Added fine-grained `record_function` spans in `modules/mamba/selective_scan_interface.py`: `ss_softplus_discretize`, `ss_state_transition_exp`, `ss_input_proj`, `ss_time_loop`, `ss_output_post` for Instruments.
- Runs:
  - CPU-grad forced (dev-clean; 120 steps): throughput≈1912 fps; loss: 359.32 → 77.05 → 95.18 → 108.89; align p50≈3,771 (T' p50≈127; U p50≈30); backend=100% cpu_grad.
    - Artifacts: `logs/rnnt_cpu_grad_120_new.csv`, `checkpoints/rnnt_cpu_grad_120_new.pt`.
  - Auto-backend (dev-clean; 150 steps): torchaudio selected but per-batch length mismatches → CPU-grad mapping; throughput≈1927 fps; loss: 412.52 → 85.47 → 131.69 → 62.69; align p50≈3,757; backend=100% cpu_grad.
    - Artifacts: `logs/rnnt_auto_150_new.csv`, `checkpoints/rnnt_auto_150_new.pt`.
- RNNT backend attempt: `pip install --no-build-isolation warp_rnnt` failed with "CPU version is not implemented" (expected). Staying with torchaudio + CPU-grad mapping path for stability on Apple Silicon.

##### 2025-08-19 (resume)
- Re-confirmed default `--max_align 60000` is appropriate for dev-clean slice; p50 align ~3.6k–4.0k, U<=40 under current caps.
- Action: propagate manual timing (CFAbsoluteTimeGetCurrent) into `MambaASRRunner` for owned telemetry during streaming evals.
- Next: run 10s streaming latency sweeps for KD/QAT/Pruned with `compute=cpu`, chunk=256; append CSVs under `exports/` and summarize here.

### Extended RNNT + Swift compute modes (today)
- RNNT CPU-grad (dev-clean; 300 steps; forced): throughput≈2115 fps; loss 360.37 → 88.04 → 84.28 → 51.78; align p50≈3,642; T' p50≈126; U p50≈28; backend=100% cpu_grad.
  - Artifacts: `logs/rnnt_cpu_grad_300_new.csv`, `checkpoints/rnnt_cpu_grad_300_new.pt`.
- RNNT CPU-grad (dev-clean; 600 steps; forced): throughput≈1704 fps; loss 410.79 → 67.56 → 53.55 → 47.32; align p50≈3,432; T' p50≈132; U p50≈26; backend=100% cpu_grad.
  - Artifacts: `logs/rnnt_cpu_grad_600_new.csv`, `checkpoints/rnnt_cpu_grad_600_new.pt`.
- Swift streaming (10s; warmup=2; real WAV) across compute:
  - all: avg≈17.75 ms; p50≈17.54; p90≈18.42; n=8
  - cpuOnly: avg≈4.05 ms; p50≈4.02; p90≈4.20; n=8
  - cpuAndGPU: avg≈18.32 ms; p50≈18.38; p90≈18.67; n=8
- Note: Small-shape Core ML path appears on CPU fast path with very low latency; ANE visibility still to be confirmed via Activity Monitor for stateful variants.

### 30s streaming CSV + 800-step RNNT (today)
- Swift streaming 30s (warmup=2; real WAV):
  - all: avg≈23.36 ms; p50≈23.18; p90≈25.64; n=8 → `exports/latency_compute_all_30s.csv`
  - cpuOnly: avg≈6.84 ms; p50≈6.66; p90≈7.74; n=8 → `exports/latency_compute_cpu_30s.csv`
  - cpuAndGPU: avg≈23.52 ms; p50≈22.32; p90≈25.05; n=8 → `exports/latency_compute_cpugpu_30s.csv`
- RNNT CPU-grad (dev-clean; 800 steps; forced): throughput≈1520 fps; loss 331.76 → 52.00 → 96.49 → 80.87; align p50≈4,342; T' p50≈129; U p50≈32; backend=100% cpu_grad.
  - Artifacts: `logs/rnnt_cpu_grad_800_new.csv`, `checkpoints/rnnt_cpu_grad_800_new.pt`.
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

##### 2025-08-19 CPU streaming latency (chunk=256, warmup=2, duration=10s, compute=cpu)
- Commands:
  - `Mamba-ASR-MPS/swift/MambaASRRunner/.build/arm64-apple-macosx/release/MambaASRRunner --mlpackage Mamba-ASR-MPS/exports/MambaASR_opt.mlpackage --stream --duration 10 --warmup 2 --compute cpu --latency-csv Mamba-ASR-MPS/exports/latency_cpu_c256_opt.csv`
  - `Mamba-ASR-MPS/swift/MambaASRRunner/.build/arm64-apple-macosx/release/MambaASRRunner --mlpackage Mamba-ASR-MPS/exports/MambaASR_opt2.mlpackage --stream --duration 10 --warmup 2 --compute cpu --latency-csv Mamba-ASR-MPS/exports/latency_cpu_c256_opt2.csv`
  - `Mamba-ASR-MPS/swift/MambaASRRunner/.build/arm64-apple-macosx/release/MambaASRRunner --mlpackage Mamba-ASR-MPS/exports/MambaASR_opt_w8.mlpackage --stream --duration 10 --warmup 2 --compute cpu --latency-csv Mamba-ASR-MPS/exports/latency_cpu_c256_w8.csv`
- Results:
  - opt: avg≈4.17 ms; p50≈4.11; p90≈4.23; n=4 → `exports/latency_cpu_c256_opt.csv`
  - opt2: avg≈4.11 ms; p50≈4.10; p90≈4.17; n=4 → `exports/latency_cpu_c256_opt2.csv`
  - w8: avg≈4.15 ms; p50≈3.94; p90≈4.31; n=4 → `exports/latency_cpu_c256_w8.csv`
  - kd: avg≈3.30 ms; p50≈3.15; p90≈3.47; n=4 → `exports/latency_cpu_c256_kd.csv`
  - qat: avg≈3.65 ms; p50≈3.47; p90≈3.78; n=4 → `exports/latency_cpu_c256_qat.csv`
  - pruned: avg≈3.37 ms; p50≈3.39; p90≈3.39; n=4 → `exports/latency_cpu_c256_pruned.csv`

##### 2025-08-19 CPU chunk-size sweep (c=128,256,512; 10s; warmup=2)
- opt: c128≈4.30 → `exports/latency_cpu_c128_opt.csv`; c256≈4.17; c512≈4.06 → `exports/latency_cpu_c512_opt.csv`
- opt2: c128≈4.22 → `exports/latency_cpu_c128_opt2.csv`; c256≈4.11; c512≈4.12 → `exports/latency_cpu_c512_opt2.csv`
- w8: c128≈4.29 → `exports/latency_cpu_c128_w8.csv`; c256≈4.15; c512≈4.21 → `exports/latency_cpu_c512_w8.csv`
- kd: c128≈3.36 → `exports/latency_cpu_c128_kd.csv`; c256≈3.30; c512≈3.43 → `exports/latency_cpu_c512_kd.csv`
- qat: c128≈3.09 → `exports/latency_cpu_c128_qat.csv`; c256≈3.65; c512≈3.32 → `exports/latency_cpu_c512_qat.csv`
- pruned: c128≈3.43 → `exports/latency_cpu_c128_pruned.csv`; c256≈3.37; c512≈3.25 → `exports/latency_cpu_c512_pruned.csv`

##### 2025-08-19 Real-audio latency + transcript sanity (QAT)
- Greedy (beam=1), CPU, chunk=256: avg≈3.62 ms; n=8 → `exports/latency_cpu_c256_qat_real.csv`
- Beam=3, CPU, chunk=256: avg≈3.34 ms; n=8 → `exports/latency_cpu_c256_qat_real_beam3.csv`
- Vocab: temporary `exports/vocab.json` (a–z cycling placeholder) used for transcript visualization.

##### 2025-08-19 Real-audio latency + transcript sanity (KD/Pruned)
- KD: greedy avg≈(see CSV) → `exports/latency_cpu_c256_kd_real.csv`; beam=3 → `exports/latency_cpu_c256_kd_real_beam3.csv`
- Pruned: greedy avg≈(see CSV) → `exports/latency_cpu_c256_pruned_real.csv`; beam=3 → `exports/latency_cpu_c256_pruned_real_beam3.csv`
- Transcripts captured to:
  - KD: `exports/transcript_kd_greedy.txt`, `exports/transcript_kd_beam3.txt`
  - Pruned: `exports/transcript_pruned_greedy.txt`, `exports/transcript_pruned_beam3.txt`

##### 2025-08-19 ALL/CPU-GPU streaming latency (chunk=256, warmup=2, duration=10s)
- all:
  - opt: avg≈19.68 ms; p50≈19.35; p90≈19.52; n=4 → `exports/latency_all_c256_opt.csv`
  - opt2: avg≈18.76 ms; p50≈18.61; p90≈18.89; n=4 → `exports/latency_all_c256_opt2.csv`
  - w8: avg≈18.56 ms; p50≈18.36; p90≈18.75; n=4 → `exports/latency_all_c256_w8.csv`
  - kd: avg≈23.84 ms; p50≈15.20; p90≈19.43; n=4 → `exports/latency_all_c256_kd.csv`
  - qat: avg≈26.54 ms; p50≈24.94; p90≈30.11; n=4 → `exports/latency_all_c256_qat.csv`
  - pruned: avg≈20.57 ms; p50≈14.88; p90≈17.17; n=4 → `exports/latency_all_c256_pruned.csv`
- cpu-gpu:
  - opt: avg≈18.63 ms; p50≈18.57; p90≈18.83; n=4 → `exports/latency_cpu-gpu_c256_opt.csv`
  - opt2: avg≈19.68 ms; p50≈19.69; p90≈19.72; n=4 → `exports/latency_cpu-gpu_c256_opt2.csv`
  - w8: avg≈18.69 ms; p50≈18.49; p90≈18.93; n=4 → `exports/latency_cpu-gpu_c256_w8.csv`
  - kd: avg≈15.70 ms; p50≈15.78; p90≈16.02; n=4 → `exports/latency_cpu-gpu_c256_kd.csv`
  - qat: avg≈21.48 ms; p50≈15.94; p90≈16.90; n=4 → `exports/latency_cpu-gpu_c256_qat.csv`
  - pruned: avg≈22.73 ms; p50≈14.55; p90≈29.81; n=4 → `exports/latency_cpu-gpu_c256_pruned.csv`

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
\n- Ran latency sweep (modes=all,cpu,cpu-gpu, chunks=256). See exports/CoreMLTraces/latency_sweep.md
