---
name: write-notes
description: Write or update repo notes under README/Notes (or the two legacy top-level note files). Use when the user wants debugging notes, investigation notes, audit notes, or institutional memory captured in the repo. Prefer updating the right high-level notes document over creating a fresh file for every session. Do not use for plans, README guides, or inline code comments.
---

# Write Notes

## Purpose

Capture useful institutional memory without creating note sprawl.

## Use When

- The user wants a bug, investigation, audit, or debugging trail captured in
  repo notes.
- A change should leave behind durable troubleshooting context.
- An existing notes bundle should be updated with a new issue section.

## Do Not Use When

- The user wants a plan (use `create-plan`).
- The output belongs in a guide or README instead (use `markdown` /
  `guide-ingest`).
- The content is really inline code documentation (use `documentation`).
- The note would be throwaway session chatter with no lasting value.

## Where notes live

Check these homes in order before creating anything new:

| Domain | File |
| --- | --- |
| Training runs, loss curves, benchmarks, MPS behavior | [README/training-notes.md](../../../README/training-notes.md) |
| Core ML export, ANE placement, telemetry | [README/coreml-telemetry-issues.md](../../../README/coreml-telemetry-issues.md) |
| Anything else, by topic | `README/Notes/<topic>.md` |

The two files above sit at the **top level of `README/`**, not inside
`README/Notes/`. That is deliberate:
[scripts/report_phase3.sh](../../../scripts/report_phase3.sh) reads
`README/training-notes.md` at runtime. **Append to it; never move or rename
it.**

## Procedure

1. Read [references/index.md](references/index.md) first.
2. Scan the two legacy notes above **and** `README/Notes/` for the best existing
   home before creating any new file.
3. Default to consolidation:
   - update the right high-level domain note file
   - add a new issue section inside that file
   - keep the issue self-contained with the
     [Notes-template](../../../README/Templates/Notes-template.md) structure
4. Create a new file under `README/Notes/` only when the topic is durable enough
   to deserve its own entry point:
   - recurring subsystem problem
   - cross-cutting audit
   - report likely to be searched directly later
5. Use topic-based names for new files (`selective-scan-mps.md`, not
   `2026-07-26-debugging.md`). Do not create a new file just because there is a
   new date or session.
6. Keep the note high signal:
   - summary
   - symptom
   - root cause or `TBD`
   - related guides
   - fix or current status
   - verification or next step
7. Keep active issues near the top. Convert fixed issues to resolved status and
   prune dead investigation branches after resolution.

## Verification sections must be honest

This repo has **no test suite**. A note whose Verification section says "tests
pass" is wrong and will mislead the next reader. Record the command that
actually ran and what it produced:

```bash
PYTHONPATH="$PWD" PYTORCH_ENABLE_MPS_FALLBACK=1 python3 train_CTC.py --epochs 1 --sanity
```

If nothing was verified, write **`Unverified`** and say why. That is a useful
note. A false green is not.

## References

Read [references/index.md](references/index.md) first.

## Handoff Rules

- Hand off to `markdown` when the real task is markdown cleanup rather than note
  placement or note structure.
- Hand off to `create-plan` when the user wants a real implementation plan
  instead of notes.
- Hand off to `guide-ingest` when the material is externally researched
  reference content, not a local investigation trail.

Never write a plan execution log here. Phase progress belongs only in the
plan's task checkboxes; the plan header states only its overall lifecycle
(`Planned`, `In-Progress`, or `Complete`). Routine test and review output is
transient; Git and CI are the evidence. Create a separate artifact only for an
important external fact that cannot be reproduced from the commit. Store it
under `README/Notes/receipts/plan-NNN/`.
