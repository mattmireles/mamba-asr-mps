---
name: coreml-profile
description: Profile Core ML model execution for mamba-asr-mps to determine which compute units (ANE, GPU, CPU) are actually being used, detect silent fallback, and identify performance bottlenecks. Triggered by keywords like "profile", "which compute unit", "is ANE running", "why is it slow", "silent fallback".
---

# Core ML Profile

## Purpose

Answer the question: **"Where is my model actually running?"** Detect silent
fallback, measure per-config compute unit utilization, and identify performance
bottlenecks across ANE/GPU/CPU on Apple Silicon.

## Use When

- The user asks which compute units a Core ML model is using.
- Performance is unexpectedly slow and the user suspects silent fallback.
- The user wants to compare `all` vs `cpu-gpu` vs `cpu`.
- After exporting a new model and wanting to verify ANE utilization.
- The user says "profile", "is ANE running", "which compute unit", "silent
  fallback".

## Do Not Use When

- The user wants numerical correctness → use
  [`coreml-validate`](../coreml-validate/SKILL.md).
- The user wants a full latency sweep and a written comparison → use
  [`bakeoff`](../bakeoff/SKILL.md).
- The user wants to export or convert a model — different workflow.

## Prerequisites

- Xcode installed and selected:
  `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`
- `powermetrics` requires `sudo` — it will prompt the user. **Do not run it
  without saying so first.**
- An exported `.mlpackage` or compiled `.mlmodelc` (default: under `exports/`).
- For the Swift path, the runner built:
  `swift build -c release --package-path swift/MambaASRRunner`

## Reference Material

- [CLAUDE.md](../../../CLAUDE.md) Part 5 — Validate → Profile → Iterate, with the
  LLDB symbolic breakpoints and `powermetrics` recipe.
- [README/Mamba-on-Apple-Silicon.md](../../../README/Mamba-on-Apple-Silicon.md) —
  MPS/ANE strategy and performance targets.
- [README/coreml-telemetry-issues.md](../../../README/coreml-telemetry-issues.md)
  — this repo's institutional memory on telemetry and placement problems. **Read
  this first**; the answer to "why is ANE not being used" may already be in it.

## Key Concepts

### Silent Fallback

`.all` is a **request**, not enforcement. Core ML may silently route ops to
CPU/GPU even when ANE is requested. The only proof is runtime telemetry.

### Why Mamba is hard on ANE

Two structural reasons specific to this model — expect partial placement and
budget for it:

1. **The selective scan is a sequential recurrence.** ANE is a throughput engine
   for dense parallel math. A scan with data-dependent per-timestep state is the
   least ANE-friendly shape there is. If it lands anywhere, it lands via an
   associative-scan reformulation — check what
   [modules/mamba/](../../../modules/mamba) actually emits.
2. **Stateful Core ML models carry state across calls.** Each `predict()` may
   force a state read/write boundary. If the graph is partitioned around that
   boundary, you pay a transition per chunk, and small chunks amplify it.

A model that is 100% ANE-resident is not the goal here. **The goal is knowing
where it runs and whether the partition is costing more than it saves.**

### Graph Partitioning Overhead

Each ANE ↔ CPU/GPU transition costs roughly 0.1–0.5 ms of context switching. A
model with many small ANE segments interleaved with CPU fallback can be
**slower** than pure CPU/GPU. This is why `cpu-gpu` beating `all` is a common
and meaningful result — not a bug.

### ANE Memory Layout Rule

The ANE pads the last axis to a multiple of 64. With `feat_dim=80` and
`d_model=256`, a `(B, T, C)` layout puts a small dimension last and wastes
bandwidth. Optimal is `(Batch, Channels, 1, SequenceLength)`. Check what the
export actually produces before concluding the model is simply "ANE-hostile."

## Procedure

### 1. Read the notes first

```bash
grep -n -i "ane\|compute unit\|fallback\|espresso" README/coreml-telemetry-issues.md
```

Thirty seconds, and it may end the investigation.

### 2. Quick power check (Level 0 — 30 seconds)

The fastest signal. **This needs `sudo` — tell the user before running it.**

Terminal 1:

```bash
sudo powermetrics -i 1000 --samplers ane -n 10
```

Terminal 2 — sustain inference long enough to register:

```bash
swift/MambaASRRunner/.build/arm64-apple-macosx/release/MambaASRRunner \
  --mlmodelc exports/Compiled_fp16_w8/MambaASR_fp16_w8.mlmodelc \
  --wav exports/tts_real_long_16k.wav \
  --stream --duration 15 --warmup 2 --compute all --chunk 256
```

**Interpretation:**

- `ANE Power: 0 mW` throughout → **silent fallback confirmed**.
- `ANE Power: >0 mW` → ANE is doing *something*, possibly only part of the
  graph.

### 3. Compute unit comparison (Level 1 — 2 minutes)

The Swift runner is the right harness — it exercises the same streaming path the
app will use, not a synthetic single-shot `predict()`.

```bash
scripts/run_latency_probe.sh exports/Compiled_fp16_w8/MambaASR_fp16_w8.mlmodelc \
  exports/tts_real_long_16k.wav
```

Defaults sweep `all`, `cpu`, `cpu-gpu` at chunk 256, 10 s duration, 2 warmup.
Results land in `exports/CoreMLTraces/latency_sweep.md`.

Override the matrix with env vars:

```bash
LAT_SWEEP_CHUNKS="128,256,512" LAT_SWEEP_MODES="all,cpu-gpu" LAT_DURATION=30 \
  scripts/run_latency_probe.sh
```

**Note the gap:** `--compute` accepts only `all`, `cpu`, and `cpu-gpu` (see
`main.swift` around the `configuration.computeUnits` switch). There is **no
`cpu-ane` option**, so you cannot directly force ANE-or-nothing from the CLI. To
get that isolation, either add the case to the runner or drive
`ct.ComputeUnit.CPU_AND_NEURAL_ENGINE` from Python. Say so rather than reporting
a comparison you could not run.

**Interpretation:**

- `cpu-gpu` faster than `all` → ANE placement is costing more than it saves
  (partitioning overhead). Ship `cpu-gpu`.
- `all` ≈ `cpu` → nothing is being offloaded at all.
- Latency scales linearly with chunk size → compute-bound. Flat → overhead-bound,
  and larger chunks are free wins.

### 4. Python-side comparison (when you need `CPU_AND_NEURAL_ENGINE`)

```bash
PYTHONPATH="$PWD" python3 -c "
import coremltools as ct, numpy as np, time, sys
MODEL = sys.argv[1]
spec = ct.models.MLModel(MODEL).get_spec()
inputs = {i.name: np.random.randn(*list(i.type.multiArrayType.shape)).astype(np.float32)
          for i in spec.description.input}
for name, cu in [('ALL', ct.ComputeUnit.ALL),
                 ('CPU_AND_GPU', ct.ComputeUnit.CPU_AND_GPU),
                 ('CPU_AND_NEURAL_ENGINE', ct.ComputeUnit.CPU_AND_NEURAL_ENGINE),
                 ('CPU_ONLY', ct.ComputeUnit.CPU_ONLY)]:
    m = ct.models.MLModel(MODEL, compute_units=cu)
    for _ in range(3): m.predict(inputs)          # warm each config separately
    ts = []
    for _ in range(11):
        t0 = time.perf_counter(); m.predict(inputs); ts.append((time.perf_counter()-t0)*1000)
    ts.sort()
    print(f'{name:24s} median={ts[5]:7.2f}ms  min={ts[0]:7.2f}  max={ts[-1]:7.2f}')
" exports/YOUR_MODEL.mlpackage
```

This does **not** exercise state threading — treat it as compute-unit triage
only, and confirm conclusions on the streaming Swift path.

### 5. xctrace profiling (Level 2 — 5 minutes)

For per-op compute unit attribution:

```bash
xctrace record --template "Core ML" \
  --output /tmp/mamba_profile.trace --time-limit 20s \
  --launch -- swift/MambaASRRunner/.build/arm64-apple-macosx/release/MambaASRRunner \
    --mlmodelc exports/Compiled_fp16_w8/MambaASR_fp16_w8.mlmodelc \
    --wav exports/tts_real_long_16k.wav \
    --stream --duration 15 --compute all --chunk 256
```

```bash
open /tmp/mamba_profile.trace
```

Look for:

- **Neural Engine track**: sustained activity = ANE is hot.
- **Gaps in the ANE track aligned to chunk boundaries** = the state boundary is
  forcing a transition every chunk. That is the Mamba-specific failure mode.
- **Thread names**: `H11ANEServicesThread` (ANE), `Espresso::MPSEngine` (GPU),
  `Espresso::BNNSEngine` (CPU).

### 6. Definitive proof via LLDB (Level 3)

Per [CLAUDE.md](../../../CLAUDE.md) Part 5 — if a breakpoint hits, you have
proof, not inference:

```text
br set -n "_ANEModel program"
br set -n "Espresso::BNNSEngine::convolution_kernel::__launch"
br set -n "Espresso::MPSEngine::context::__launch_kernel"
```

## Output Template

```text
## Core ML Profile: [model name]

### Machine
- Chip: [e.g. M2 Pro] | RAM: [..] | macOS: [..] | On AC power: [yes/no]

### Prior art
- README/coreml-telemetry-issues.md: [relevant entry / none found]

### Power Check
- ANE Power during inference: [X mW / 0 mW]
- Verdict: [ANE active / silent fallback / partial]

### Compute Unit Comparison (streaming, chunk=256, 10s, 2 warmup)
| Config   | p50 (ms) | p90 | p99 |
| all      |          |     |     |
| cpu-gpu  |          |     |     |
| cpu      |          |     |     |

Source: exports/CoreMLTraces/latency_sweep.md

### Not measured
- CPU_AND_NEURAL_ENGINE in isolation: the runner has no `cpu-ane` mode.

### Fastest Config: [X] ([Y]× faster than `all`)

### Recommendation
- ...
```

## Anti-patterns

- **Assuming `.all` means ANE** — it doesn't. Verify with telemetry.
- **Profiling without warmup** — the first prediction includes ANE plan
  compilation. The runner's `--warmup` exists for this; use at least 2.
- **Profiling on battery** — thermal and power throttling skew results. Plug in
  and say so in the report.
- **Single-iteration timing** — use the streaming duration sweep, report p50/p90/p99.
- **Reporting a `cpu-ane` comparison you couldn't run** — the CLI has no such
  mode. State the gap.
- **Treating partial ANE placement as failure** — for a sequential SSM,
  partial is the realistic ceiling. Judge by measured latency, not by ANE
  residency percentage.
- **Running `sudo powermetrics` without telling the user first.**

## Canonical References

- [CLAUDE.md](../../../CLAUDE.md) Part 4 (ANE layout) and Part 5 (profiling ladder)
- [README/coreml-telemetry-issues.md](../../../README/coreml-telemetry-issues.md)
- [README/Mamba-on-Apple-Silicon.md](../../../README/Mamba-on-Apple-Silicon.md)
- [scripts/run_latency_probe.sh](../../../scripts/run_latency_probe.sh),
  [scripts/summarize_latency_csv.py](../../../scripts/summarize_latency_csv.py)
- `swift/MambaASRRunner/Sources/*/main.swift` — the compute-mode switch
- Sibling skills: [`coreml-validate`](../coreml-validate/SKILL.md),
  [`bakeoff`](../bakeoff/SKILL.md), [`debug`](../debug/SKILL.md)
