---
name: bakeoff
description: Run the controlled latency and accuracy bakeoff on this machine for mamba-asr-mps. Prepares inputs, builds the Swift runner, sweeps compute modes and chunk sizes, records results, and updates README/training-notes.md. Use when the user says "run the bakeoff", "benchmark this machine", "compare the configs", or invokes $bakeoff.
---

# Bakeoff

## Purpose

Produce a **defensible** performance comparison across model variants, compute
modes, and chunk sizes on **this specific machine**, and record it where the next
person will find it.

A bakeoff is not "run it twice and eyeball the numbers." It is a controlled
measurement whose result someone else can act on six months from now.

## Use When

- The user says "run the bakeoff", "benchmark this machine", "compare configs",
  "is the optimized model actually faster".
- After a quantization, pruning, or distillation pass in
  [scripts/optimize.py](../../../scripts/optimize.py), to prove the win.
- Before a handoff, to fill in the latency numbers the
  [`deploy`](../deploy/SKILL.md) contract requires.

## Do Not Use When

- The question is **where** the model runs → [`coreml-profile`](../coreml-profile/SKILL.md).
- The question is **whether the export is correct** →
  [`coreml-validate`](../coreml-validate/SKILL.md).
- The user wants a single quick timing, not a controlled sweep.

## Hygiene — read before running anything

Measurement discipline is the whole value of this skill. Violate these and the
numbers are noise:

| Rule | Why |
| --- | --- |
| **Plug in to AC power** | Apple Silicon throttles aggressively on battery. Battery numbers are not comparable to anything. |
| **Warm up, always** | First inference includes Core ML plan compilation and ANE program load. The runner's `--warmup` defaults to 2; use ≥2. |
| **Quit heavy apps** | Xcode indexing, browsers, and Docker all steal GPU/ANE and thermal headroom. |
| **Let the machine cool between sweeps** | A sweep run right after a training job measures a hot machine. |
| **Report p50/p90/p99, never the mean alone** | Streaming ASR lives and dies on tail latency. A good mean with a bad p99 drops words. |
| **Counterbalance the order** | Run configs A→B and B→A. If order changes the ranking, thermal drift is contaminating the result — extend cooldowns. |
| **Record the machine** | Chip, RAM, macOS version, on-AC. A number without a machine is not a result. |
| **Same audio, same chunking, every config** | Vary exactly one thing at a time. |

## Procedure

### 1. Record the environment

```bash
{ sysctl -n machdep.cpu.brand_string
  echo "RAM: $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"
  sw_vers -productVersion
  pmset -g batt | head -1
  python3 -c "import torch; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available())"
  git rev-parse --short HEAD
} 2>&1
```

Paste this verbatim into the results. If `pmset` does not say `AC Power`, stop
and plug in.

### 2. Build the runner in release

Debug builds are meaningless for latency.

```bash
swift build -c release -Xswiftc -O --package-path swift/MambaASRRunner
```

`run_latency_probe.sh` builds it automatically if missing, but build explicitly
so a compile failure surfaces as a compile failure, not as a mid-sweep gap.

### 3. Confirm inputs exist

```bash
ls -la exports/*.mlmodelc exports/Compiled_*/*.mlmodelc exports/*.wav 2>/dev/null
```

Defaults the harness expects:

- Model: `exports/Compiled_fp16_w8/MambaASR_fp16_w8.mlmodelc`
- Audio: `exports/tts_real_long_16k.wav`

Missing? Export first — `python3 scripts/export_and_validate.py` — and note in
the results that the artifact was regenerated for this run.

Use **long-form real audio**. A 3-second clip cannot fill a 256-frame chunk
pipeline or expose state drift.

### 4. Run the sweep

```bash
LAT_SWEEP_MODES="all,cpu,cpu-gpu" \
LAT_SWEEP_CHUNKS="128,256,512" \
LAT_DURATION=30 LAT_WARMUP=3 \
LAT_SWEEP_TAG="fp16_w8" \
scripts/run_latency_probe.sh
```

Outputs land in `exports/CoreMLTraces/`:

- `latency_<mode>_c<chunk>_<tag>.csv` — raw per-chunk timings
- `latency_<mode>_c<chunk>_<tag>.md` — per-config summary
- `latency_sweep_<tag>.md` — combined report

The harness tolerates individual config failures (`|| true`) and writes
`(no CSV produced for <tag>)` instead. **Check for that line** — a silently
skipped config reads as a missing row, not an error.

### 5. Counterbalance

Re-run with the mode order reversed:

```bash
LAT_SWEEP_MODES="cpu-gpu,cpu,all" LAT_SWEEP_TAG="fp16_w8_rev" \
LAT_DURATION=30 LAT_WARMUP=3 scripts/run_latency_probe.sh
```

If the winner changes between the two orders, the result is thermal, not real.
Cool down and rerun with longer warmup before reporting anything.

### 6. Compare model variants

```bash
for m in base opt opt2 w8; do
  LAT_SWEEP_TAG="$m" scripts/run_latency_probe.sh "exports/Compiled_${m}/MambaASR_${m}.mlmodelc"
done
python3 scripts/compare_sweeps.py
```

[scripts/compare_models_cpu.py](../../../scripts/compare_models_cpu.py) reads the
CPU-only table straight out of `README/implementation-plan-v2.md` — that plan
carries the historical baseline. Compare against it rather than re-deriving one.

### 7. Latency is only half the bakeoff

A faster model that transcribes worse is not a win. Pair every latency number
with accuracy on the same artifact:

```bash
python3 scripts/compute_wer_cer.py
```

Report latency and WER/CER **together**. A config row without a WER is
incomplete.

### 8. Record the results

Append to [README/training-notes.md](../../../README/training-notes.md) via the
[`write-notes`](../write-notes/SKILL.md) skill — that file is the repo's
benchmark memory, and
[scripts/report_phase3.sh](../../../scripts/report_phase3.sh) reads it.

Include: the environment block from step 1, the results table, the
counterbalance check, WER/CER, and **what you did not measure**.

## Output Template

```text
## Bakeoff — [date]

### Machine
- Chip: ... | RAM: ... GB | macOS: ... | Power: AC
- torch: ... | MPS available: ... | commit: ...

### Artifacts
- Model(s): ... (regenerated this run: yes/no)
- Audio: ... (duration: ...s)

### Results (p50 / p90 / p99 ms per chunk)
| Model | Mode    | Chunk | p50 | p90 | p99 | WER |
| ----- | ------- | ----- | --- | --- | --- | --- |

### Counterbalance
- Forward order winner: ...
- Reverse order winner: ...
- Consistent: [yes / NO — result is thermal, do not trust]

### Verdict
- Fastest config: ... (...× vs baseline), WER ...%
- Recommendation: ...

### Not measured
- ...
```

## Anti-patterns

- **Benchmarking on battery.** The single most common way to produce a wrong
  number on a Mac.
- **No warmup.** You measured Core ML compiling the model.
- **Reporting a mean.** Report p50/p90/p99; streaming ASR is a tail-latency
  problem.
- **One order only.** Without a counterbalance you cannot distinguish a real
  ranking from thermal drift.
- **Latency without WER.** A quantized model that got 2× faster and 4 points
  worse is a regression being reported as a win.
- **Ignoring `(no CSV produced ...)`.** A skipped config is a hole in the
  result, not an absent row.
- **Not recording the machine.** An unattributed number cannot be compared to
  anything later, which makes the whole run worthless.
- **Comparing against a number from a different machine or commit** without
  saying so.

## Related skills

- [`coreml-profile`](../coreml-profile/SKILL.md) — *where* it runs (use this
  first if a config is unexpectedly slow).
- [`coreml-validate`](../coreml-validate/SKILL.md) — is the export even correct.
- [`write-notes`](../write-notes/SKILL.md) — recording results.
- [`deploy`](../deploy/SKILL.md) — the handoff contract these numbers feed.
