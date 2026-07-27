---
name: coreml-validate
description: Validate Core ML model numerical correctness against the PyTorch reference for mamba-asr-mps. Runs identical inputs through both pipelines and reports per-output correlation, max error, and pass/fail. Triggered by keywords like "validate", "numerical parity", "does Core ML match PyTorch", "correlation", "drift", "did the export break it".
---

# Core ML Validate

## Purpose

Answer the question: **"Does the Core ML model produce the same output as
PyTorch?"** Run identical inputs through both pipelines and report per-output
tensor correlation, max absolute error, and pass/fail against defined
thresholds.

## Use When

- After exporting a new Core ML model and wanting to verify numerical parity.
- After changing export settings (`compute_precision`,
  `minimum_deployment_target`, quantization).
- Debugging transcript quality issues that might be conversion artifacts rather
  than model quality.
- The user says "validate", "does it match", "numerical parity", "correlation",
  "drift".
- Comparing FP16 vs FP32 Core ML output against PyTorch FP32.

## Do Not Use When

- The user wants to know **which compute units** are being used → use
  [`coreml-profile`](../coreml-profile/SKILL.md).
- The user wants a **latency comparison** → use [`bakeoff`](../bakeoff/SKILL.md).
- The user wants **transcription accuracy** → that is WER/CER via
  `scripts/compute_wer_cer.py`, not tensor parity.
- The model hasn't been exported yet — export first, then validate.

## Gate

**Stop** unless both exist:

1. A Core ML `.mlpackage` (or compiled `.mlmodelc`) for the model under test.
   Default export output lands under `exports/`.
2. A PyTorch reference — a checkpoint loadable by `modules/Conmamba.py` or
   `modules/mct/`.

If no `.mlpackage` exists yet, run the export first:

```bash
python3 scripts/export_coreml.py
```

## Prerequisites

- `coremltools`, `torch`, `numpy` in the environment (`requirements.txt`).
- Both PyTorch checkpoint and the `.mlpackage` available.
- `PYTHONPATH="$PWD"` so `modules/` and `utils/` import.

## Reference Material

Before validating, consult:

- [CLAUDE.md](../../../CLAUDE.md) Part 4 — Core ML / ANE edge cases (FP16 drift,
  memory layout, stateful caches).
- [CLAUDE.md](../../../CLAUDE.md) Part 5 — Validate → Profile → Iterate ladder.
- [README/coreml-telemetry-issues.md](../../../README/coreml-telemetry-issues.md)
  — what has already broken in this repo's export path.
- [scripts/export_coreml.py](../../../scripts/export_coreml.py) — the actual
  conversion settings, state definitions, and ANE targeting in use.

## What makes this model hard to validate

The export is **stateful**. Mamba's hidden state is declared as a Core ML
`StateType` so the runtime carries recurrence across streaming chunks. Two
consequences:

1. **State is not an input you can randomize.** A single-shot `predict()` with
   random inputs validates the *chunk* transform, not the recurrence. To
   validate the recurrence you must feed a **sequence** of chunks in order and
   compare the trajectory, keeping the PyTorch reference's state threaded the
   same way.
2. **State drift compounds.** A tiny per-chunk FP16 error that looks fine at
   chunk 1 can dominate by chunk 50. Always validate over a realistic chunk
   count, not one call.

Validate in this order — first divergence wins:

| Stage | Isolate |
| --- | --- |
| 1. Frontend | Conv subsampling output, single chunk, no state |
| 2. Single Mamba block | One chunk, zero-initialized state |
| 3. Full encoder, one chunk | Zero-initialized state |
| 4. Full encoder, N chunks | Threaded state — this is where drift shows up |
| 5. CTC head / decode | Logits, then the argmax/beam path |

## Procedure

### 1. Identify what to validate

- Which model? (path to `.mlpackage` under `exports/`)
- Against what reference? (PyTorch eager, or the traced module)
- What precision? (FP16 default, FP32 for isolating quantization)
- What input? **Real mel features beat random noise** — random inputs miss
  preprocessing and normalization mismatches. Use a real clip via
  `datasets/librispeech_csv.py` when available.

### 2. Inspect the exported interface first

Cheap, and catches shape and state mistakes before any comparison:

```bash
PYTHONPATH="$PWD" python3 -c "
import coremltools as ct, sys
m = ct.models.MLModel(sys.argv[1])
spec = m.get_spec()
print('inputs:')
for i in spec.description.input:
    print('   ', i.name, list(i.type.multiArrayType.shape), i.type.multiArrayType.dataType)
print('outputs:')
for o in spec.description.output:
    print('   ', o.name, list(o.type.multiArrayType.shape))
st = getattr(spec.description, 'state', [])
print('states:', [s.name for s in st] or 'NONE  <-- expected StateType for Mamba recurrence')
" exports/YOUR_MODEL.mlpackage
```

If `states` is empty on a model that should be stateful, stop — the export is
wrong and any parity number you compute afterwards is meaningless.

### 3. Isolate quantization from conversion (do this before blaming Core ML)

Compare Core ML against **itself** at two precisions first. If FP16-vs-FP32
inside Core ML already diverges, the problem is quantization, not the
conversion:

```bash
PYTHONPATH="$PWD" python3 -c "
import coremltools as ct, numpy as np, sys
MODEL = sys.argv[1]
spec = ct.models.MLModel(MODEL).get_spec()
np.random.seed(42)
inputs = {i.name: np.random.randn(*list(i.type.multiArrayType.shape)).astype(np.float32)
          for i in spec.description.input}
a = ct.models.MLModel(MODEL, compute_units=ct.ComputeUnit.ALL).predict(inputs)
b = ct.models.MLModel(MODEL, compute_units=ct.ComputeUnit.CPU_ONLY).predict(inputs)
for k in a:
    x, y = np.array(a[k]).ravel(), np.array(b[k]).ravel()
    print(f'{k}: corr={np.corrcoef(x,y)[0,1]:.6f}  max_err={np.abs(x-y).max():.6f}')
" exports/YOUR_MODEL.mlpackage
```

### 4. Compare against PyTorch

Feed the **same** tensor to both. Force the PyTorch side to CPU/FP32 so you are
measuring the export, not MPS:

```bash
PYTHONPATH="$PWD" python3 -c "
import coremltools as ct, numpy as np, torch
torch.manual_seed(0); np.random.seed(42)

# --- PyTorch reference (CPU/FP32) ---
# from modules.Conmamba import ConmambaEncoder   # or modules.mct
# model = ...load checkpoint...; model.eval().float().cpu()

x = np.random.randn(1, 256, 80).astype(np.float32)   # (B, T, feat_dim=80)

# with torch.no_grad():
#     ref = model(torch.from_numpy(x)).numpy()

cm = ct.models.MLModel('exports/YOUR_MODEL.mlpackage',
                       compute_units=ct.ComputeUnit.CPU_ONLY)
out = cm.predict({'INPUT_NAME': x})

for k, v in out.items():
    v = np.array(v).ravel()
    print(f'{k}: shape={v.shape} range=[{v.min():.4f}, {v.max():.4f}]')
    # r = np.corrcoef(v, ref.ravel())[0,1]
    # print(f'  corr={r:.6f}  max_err={np.abs(v-ref.ravel()).max():.6f}')
"
```

**Always print the range.** A model whose outputs are 1000× the reference has
perfect correlation and is completely broken.

### 5. Validate the recurrence over a chunk sequence

The step most people skip. Feed N consecutive chunks from one real utterance,
threading state on both sides, and report correlation **per chunk index** — not
one aggregate number. A clean chunk-1 correlation with a collapsing chunk-50
correlation is the signature of state drift, and it is invisible to single-shot
validation.

### 6. Confirm end-to-end through Swift

The Python `predict()` path and the Swift runtime path can disagree —
different compute-unit defaults, different state handling. Close the loop:

```bash
python3 scripts/export_and_validate.py
```

This orchestrates export → `.mlmodelc` compile → `MambaASRRunner` streaming
validation. It also emits the character vocab the Swift decoder needs —
`exports/vocab_char_29.json` by default, or wherever `--vocab_out` points.

Then check the transcript is real text, not blank-collapsed gibberish:

```bash
swift/MambaASRRunner/.build/arm64-apple-macosx/release/MambaASRRunner \
  --mlmodelc exports/Compiled_fp16_w8/MambaASR_fp16_w8.mlmodelc \
  --wav exports/tts_real_long_16k.wav \
  --vocab exports/vocab_char_29.json --compute cpu --chunk 256
```

**The vocab must match what the model was trained with.** The 29-symbol
character map generated here (blank, space, a–z, apostrophe) is the CTC
character path. A model trained against a different tokenizer
([utils/tokenizer.py](../../../utils/tokenizer.py)) will decode to confident,
fluent nonsense — high logit correlation, unusable transcript. Check the vocab
before blaming the export.

## Pass/Fail Thresholds

| Metric | Excellent | Acceptable | Investigate | Broken |
| --- | --- | --- | --- | --- |
| Pearson correlation | > 0.999 | > 0.99 | > 0.90 | < 0.90 |
| Max absolute error (FP16) | < 0.001 | < 0.01 | < 0.1 | > 0.1 |
| Max absolute error (FP32) | < 1e-5 | < 1e-4 | < 1e-3 | > 1e-3 |

**For ASR, tensor parity is necessary but not sufficient.** The metric that
decides shipping is **WER/CER on real audio**:

```bash
python3 scripts/compute_wer_cer.py
```

A model with 0.995 logit correlation can still be unusable if the drift lands on
blank-vs-token decisions near the CTC decision boundary. Conversely, a model
with mediocre correlation on padded regions can transcribe perfectly. **Report
both; let WER decide.**

## Output Template

```text
## Core ML Validation: [model name]

### Config
- Core ML: [path] ([FP16/FP32], [compute_units])
- Reference: [PyTorch eager CPU/FP32 @ commit]
- Input: [real clip / random seed 42], [N] chunks of [size]

### FP16 vs FP32 (Core ML internal)
| Output | Correlation | Max Error |

### Core ML vs PyTorch
| Output | Correlation | Max Error | Verdict |

### State drift (per chunk index)
| Chunk | Correlation |
| 1     | ...         |
| 25    | ...         |
| 50    | ...         |

### Task metric
- WER: ...%   CER: ...%   (eval set: ...)
- PyTorch reference WER: ...%

### Overall Verdict: [PASS / INVESTIGATE / FAIL]

### Recommendation
- ...
```

## Anti-patterns

- **Validating one chunk on a stateful model** — that tests the transform, not
  the recurrence, which is the part most likely to be wrong.
- **Comparing FP16 Core ML vs FP32 PyTorch and blaming "Core ML"** — first
  compare FP16 vs FP32 *inside* Core ML to isolate quantization from conversion.
- **Comparing against PyTorch on MPS** — MPS is itself a variable. Pin the
  reference to CPU/FP32.
- **Random inputs only** — real mel features catch preprocessing and
  normalization mismatches that noise never will.
- **Ignoring output scale** — check ranges; correlation is scale-invariant.
- **Reporting correlation as if it were accuracy** — for ASR, run WER.
- **Testing one utterance** — edge cases (very short, silence, long-form) expose
  state and padding bugs that typical clips don't.

## Canonical References

- [CLAUDE.md](../../../CLAUDE.md) Part 4 and Part 5
- [README/coreml-telemetry-issues.md](../../../README/coreml-telemetry-issues.md)
- [scripts/export_coreml.py](../../../scripts/export_coreml.py),
  [scripts/export_and_validate.py](../../../scripts/export_and_validate.py)
- [scripts/coreml_ops_remediation.py](../../../scripts/coreml_ops_remediation.py)
  — op-level fixes already applied here
- Sibling skills: [`coreml-profile`](../coreml-profile/SKILL.md),
  [`bakeoff`](../bakeoff/SKILL.md), [`debug`](../debug/SKILL.md)
