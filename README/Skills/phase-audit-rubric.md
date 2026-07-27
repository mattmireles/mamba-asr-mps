# Phase Audit Rubric

Canonical review checklist for `phase-audit` and the local audit fallback used
by `execute-plan`.

## Review Goal

Decide whether the current phase is actually complete, safely committed, and
ready for the next phase or push.

## Required Checks

### 1. Scope Completion

Double-check and audit all the work. Review the code.

- Did the implementation complete the exact scope promised by the current phase?
- Did we miss anything?

### 2. Checkbox Accuracy

Check the boxes and update the plan.

- Do the checked boxes in the plan match reality?
- Is anything marked complete that was only partially done?

### 3. Canonical-Guide Alignment

- Does the implementation still follow the linked `README/Guides`, `docs/`, and
  [Mamba-on-Apple-Silicon.md](../Mamba-on-Apple-Silicon.md)?
- Did it create a parallel workflow or duplicate a canonical doc by accident?

### 4. Edge Cases

Weight these by what actually breaks in this repo:

- **Shapes and lengths** — CTC output lengths after subsampling, RNN-T joint
  dimensions, padding/mask alignment.
- **Device and dtype** — anything that silently lands on CPU when it should be
  on MPS, or mixes fp16/fp32 across a boundary.
- **MPS op gaps** — does this path still work with
  `PYTORCH_ENABLE_MPS_FALLBACK=1`, and did the fallback quietly become the hot
  path?
- **SSM numerical stability** — discretization, cumulative products, and
  anything that can produce `NaN`/`inf` over long sequences.
- **Export tracing** — data-dependent control flow, dynamic shapes, or ops that
  break `coremltools.convert`.
- Did the change preserve existing guardrails and invariants?

### 5. Verification

- Was the mechanical gate run?
  `PYTHONPATH="$PWD" python3 train_CTC.py --epochs 1 --sanity`
- Were the surface-specific checks run (RNN-T sanity, export validate, SSM
  bench) when the phase touched that surface?
- Is there a meaningful verification statement for the phase — an exit code, a
  loss value, a measured latency — not a claim?
- If a check was not run, is that gap stated clearly?

**There is no test suite in this repo.** "Tests pass" is never a valid
verification statement here. Cite the actual command and its actual output.

### 6. Commit Readiness

- Is the change coherent enough to commit as one phase?
- Are there stray edits, debug leftovers, or half-finished scaffolding?
- Did any checkpoint, `.npy`, `.mlpackage`, or `swift/**/.build/` artifact leak
  into the staged set?

### 7. Push and CI Readiness

- Is the repo ready for the next push?
- Will the CPU-only `sanity-train` job in
  [ci.yml](../../.github/workflows/ci.yml) still pass? CI installs only
  [requirements-ci.txt](../../requirements-ci.txt) (torch + numpy) on
  `ubuntu-latest` — a new import of `coremltools`, `librosa`, `soundfile`, or
  `torchaudio` on the `train_CTC.py --sanity` path **will** break CI even
  though it works locally.

## Findings Format

Report findings in this order:

1. High severity findings
2. Medium severity findings
3. Low severity findings
4. Residual risks or verification gaps

If there are no findings, say so explicitly and still mention any remaining
verification gaps.

## Decision Rule

- If scope, checkbox accuracy, or safety is wrong, the phase is not complete.
- If findings require fixes, do not mark the phase complete yet.
- If no blocking findings remain, the phase can be marked complete and
  committed.
