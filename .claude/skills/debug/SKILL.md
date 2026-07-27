---
name: debug
description: >-
  Systematic debugging for mamba-asr-mps: consult README/ and docs/ (guides and
  learnings) and CLAUDE.md first, pull current library docs via Context7 MCP
  when PyTorch MPS / coremltools / API behavior is uncertain, parallelize
  investigation with multiple subagents when stuck, prove fixes before calling
  success, then capture one consolidated note via **write-notes** as the
  **final** step before ending the session. Use when the user invokes **debug**,
  **use debug**, asks to debug or fix a tricky bug, or work is blocked by
  unclear failure modes after quick local checks.
---

# Debug

## Purpose

Reduce guesswork: **repo docs and CLAUDE.md first**, **Context7 when library
behavior is uncertain**, **parallel hypotheses when stuck**, **proof before
"fixed"**, **one write-notes pass at session end**.

## Use When

- The user says **debug**, **use the debug skill**, **debug this**, or the task
  is **bug investigation** / **root-cause** analysis that is stalling.
- Failures involve **PyTorch MPS**, **`coremltools`**, **Core ML runtime**, or
  **Swift integration** where training data may be stale — use Context7 before
  assuming APIs.

## Do Not Use When

- The user only wants a **trivial fix** with an obvious stack trace and no doc
  ambiguity (still skim the guides if the area is known to be finicky).
- The task is **greenfield feature build** with no defect.

## Workflow (in order)

### 1. Read repo knowledge first (mandatory)

Before writing code or deep-diving the stack:

1. **Match the symptom to a guide:**

   | Symptom area | Read first |
   | --- | --- |
   | MPS slowness, device placement, ANE targeting | [README/Mamba-on-Apple-Silicon.md](../../../README/Mamba-on-Apple-Silicon.md) — cited by section number from `train_CTC.py`, `train_RNNT.py`, `utils/metrics.py`, `utils/tokenizer.py`, `benchmarks/bench_mps.py` |
   | Core ML export, ANE placement, telemetry | [README/coreml-telemetry-issues.md](../../../README/coreml-telemetry-issues.md) |
   | Training divergence, loss curves, benchmark history | [README/training-notes.md](../../../README/training-notes.md) |
   | Exceptions, error paths, failure handling | [docs/ERROR_HANDLING_GUIDE.md](../../../docs/ERROR_HANDLING_GUIDE.md) |
   | Module boundaries, data flow | [docs/SYSTEM_ARCHITECTURE.md](../../../docs/SYSTEM_ARCHITECTURE.md) |
   | iOS/macOS runtime integration | [docs/INTEGRATION_GUIDE.md](../../../docs/INTEGRATION_GUIDE.md) |
   | PyTorch MPS semantics, deployment | [docs/Mamba-Apple-Silicon-guide.md](../../../docs/Mamba-Apple-Silicon-guide.md) |

2. **Skim [README/Notes/](../../../README/Notes)** and the roadmaps
   ([implementation-plan-v2.md](../../../README/implementation-plan-v2.md)) for
   past investigations and what was already ruled out.
3. Read **[CLAUDE.md](../../../CLAUDE.md)** for the MPS training and
   PyTorch → Core ML playbook constraints (static shapes, ANE layout,
   divide-and-conquer).

### 2. Context7 MCP (library and platform docs)

When the bug touches **PyTorch MPS**, **`coremltools`**, **`torchaudio`**,
**Swift Core ML**, or related APIs:

1. Use **Context7**: resolve the library ID, then fetch focused docs for the API
   or behavior in question.
2. Prefer Context7 over memory or generic web search **for API shape and
   version-sensitive behavior**. MPS op coverage in particular changes between
   PyTorch minor releases — do not answer from training data.

If Context7 is unavailable, fall back to official docs — **do not** invent APIs.

### 3. Reproduce and narrow

- Confirm a **minimal repro**. The cheapest one in this repo is the synthetic
  sanity path:

  ```bash
  PYTHONPATH="$PWD" PYTORCH_ENABLE_MPS_FALLBACK=1 python3 train_CTC.py --epochs 1 --sanity
  ```

- State a **one-sentence hypothesis** and what evidence would falsify it.

### 4. Bisect by layer, not by guess

This stack has clean seams. Cut at them:

| Layer | Isolate with |
| --- | --- |
| SSM kernel numerics | `python3 benchmarks/bench_selective_scan.py`, `python3 scripts/bench_selective_scan_report.py` |
| MPS vs CPU behavior | run the same path with `PYTORCH_ENABLE_MPS_FALLBACK=1` vs forcing CPU; a bug that vanishes on CPU is a device/op bug |
| Model forward vs training loop | instantiate `modules/Conmamba.py` directly with a synthetic batch |
| Loss implementation | `modules/rnnt_loss.py` vs `modules/rnnt_loss_mps.py`; `python3 scripts/bench_rnnt_impls.py` compares them |
| PyTorch vs Core ML | `python3 scripts/compare_models_cpu.py`, `python3 scripts/export_and_validate.py` |
| Python vs Swift runtime | `swift/MambaASRRunner` against the same input |

**The single highest-value question in this repo:** does the bug survive on CPU?
If it does not, it is MPS — an op with no Metal kernel silently falling back, a
dtype the Metal path handles differently, or a synchronization gap.

### 5. Parallel investigation when stuck

If the problem is **hard**, **cross-cutting**, or you have been **going in
circles** after the reproduce-and-narrow pass:

- In **one turn**, spin up **multiple** readonly subagents with **narrow
  charters** (e.g. "SSM numerics only", "export tracing only", "Swift shapes
  only").
- **Merge** overlapping hypotheses; avoid duplicate deep reads of the same file.

**When in doubt, parallelize** (same posture as **`audit`**, but
hypothesis-driven).

### 6. Prove the fix (before claiming success)

Do **not** say **fixed** until there is **objective proof**. What counts here:

- the sanity train completes with an exit code of 0 and a finite, decreasing
  loss
- the previously-`NaN` tensor is finite across the failing input range
- `scripts/export_and_validate.py` reports correlation above threshold
- the bad log line is absent from a fresh run
- a measured latency moved in the direction claimed, on a warmed run

**"Should work" is not proof.** If you could not verify, say **unverified** and
say why.

### 7. Write notes once (mandatory last step)

**After** investigation, perform **a single** notes update using
[**`write-notes`**](../write-notes/SKILL.md) (consolidate the session).

- One pass: symptom, repro, ruled out, root cause (`TBD` if unknown), fix,
  **verification** (proven / not), pointers to guides.
- Default home for MPS and training issues is
  [README/training-notes.md](../../../README/training-notes.md); for export and
  ANE issues,
  [README/coreml-telemetry-issues.md](../../../README/coreml-telemetry-issues.md).
  New domains get a file under [README/Notes/](../../../README/Notes).
- **Skip** only when the user explicitly wants no note or the outcome is
  trivial.

## Output expectations

- **What was read** from `README/`, `docs/`, `CLAUDE.md` (paths).
- **Verification:** what proved the fix (or **unverified** / **blocked**).
- **Which note file(s)** were updated in step 7 (or why skipped).
- **Whether Context7** was used and for which topic.
- **Leading hypothesis(es)** if still open, else **root cause** summary.

## Anti-patterns

- Skipping **`README/`**, **`docs/`**, **`CLAUDE.md`** to "save time."
- Assuming PyTorch MPS op coverage without **Context7 or official docs** — it
  changes every minor release.
- Blaming "MPS being flaky" without running the CPU comparison that would
  confirm or kill it.
- **Serial** thrashing instead of **parallel** charters.
- **Declaring victory** without **verification**.
- **Patching notes throughout** instead of **one** end-of-session pass.

## Relation to other skills

- **`audit`:** structured findings-first review; escalate to it when debug maxes
  out.
- **`coreml-validate`:** when the question is specifically PyTorch↔Core ML
  numerical parity.
- **`coreml-profile`:** when the question is specifically where the model runs.
- **`write-notes`:** once at end of debug session.
- **`phase-audit`:** plan-phase checks — not the same as production debugging.
