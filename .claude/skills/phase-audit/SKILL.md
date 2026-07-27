---
name: phase-audit
description: Audit a completed plan phase in this repo. Use when a phase has just been implemented and needs a findings-first review against the active plan, changed files, linked guides, and the canonical audit rubric before commit or before moving to the next phase. Do not use for implementing fixes, broad product critique, or replacing verification runs and CI execution.
---

# Phase Audit

## Purpose

Use this skill to audit a completed implementation phase like a skeptical senior
engineer doing code review. The job is to decide whether the phase is actually
complete and ready to commit or push, not to help implement it.

## Use When

- A plan phase was just implemented.
- The current plan checkboxes need validation against reality.
- The repo needs a findings-first review before commit or before moving to the
  next phase.

## Do Not Use When

- The request is to implement or patch code.
- The request is broad product or UX feedback.
- The request is to replace CI or runtime verification with a code review.

## Procedure

1. Read [references/index.md](references/index.md) first.
2. Read the active plan and isolate the exact phase being audited:
   - phase goal
   - checked tasks
   - phase verification text
   - linked canonical references in
     [README/Skills](../../../README/Skills),
     [README/Guides](../../../README/Guides), and [docs/](../../../docs)
3. Inspect the implementation evidence:
   - changed files and diffs
   - the verification commands that were actually run, and their output
   - any notes that explain scope decisions
4. Compare the implementation against
   [phase-audit-rubric.md](../../../README/Skills/phase-audit-rubric.md) and the
   linked guides.
5. Produce a findings-first audit:
   - severity ordered
   - concrete file references when applicable
   - explicit call on whether the phase is complete
6. Explicitly check:
   - missing scope
   - missing edge cases — shapes and lengths, device/dtype, MPS op gaps, SSM
     numerical stability, export traceability
   - weak or absent verification
   - checkbox drift between the plan and the code
   - stray artifacts in the staged set (checkpoints, `.npy`, `.mlpackage`,
     `swift/**/.build/`)
   - commit readiness
   - push and CI readiness — will the CPU-only `sanity-train` job still pass
     with only `requirements-ci.txt` installed?
7. If there are no findings, say so directly and note residual risks or
   verification gaps.

## Verification reality check

The single most common false claim in this repo is "tests pass." **There are no
tests.** When a phase says it was verified, confirm which of these actually ran
and what it produced:

| Surface | Command |
| --- | --- |
| CTC (always) | `PYTHONPATH="$PWD" python3 train_CTC.py --epochs 1 --sanity` |
| RNN-T | `PYTHONPATH="$PWD" python3 train_RNNT.py --epochs 1 --sanity` |
| Core ML export | `python3 scripts/export_and_validate.py` |
| SSM kernels | `python3 benchmarks/bench_selective_scan.py` |
| MPS performance | `python3 benchmarks/bench_mps.py` |
| Swift runner | `swift build -c release --package-path swift/MambaASRRunner` |

A verification statement without a command and an observable result is a
finding, not evidence.

## Output Contract

- Findings first.
- Severity ordered.
- Concrete file references when applicable.
- Brief summary only after the findings.

## Boundaries

- Do not edit files.
- Do not silently fix the problem instead of reporting it.
- Do not replace CI or runtime verification with review prose.

## Canonical Docs

Read [references/index.md](references/index.md) first. It maps the workflow
guide, the audit rubric, and the repo artifacts that define whether a phase is
actually done.

## Handoff Rules

- Hand findings back to `execute-plan` or the user for remediation.
- Re-run this skill after fixes if the phase changed materially.
