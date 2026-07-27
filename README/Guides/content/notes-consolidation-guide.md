# Notes consolidation guide

## Purpose

Keep institutional memory in `README/Notes/` **high signal and low sprawl**.
Prefer updating an existing domain note over adding a new file whenever the
topic fits.

## Existing homes

Check these before creating anything new:

| Domain | File |
| --- | --- |
| Training runs, benchmarks, loss curves, MPS behavior | [README/training-notes.md](../../training-notes.md) |
| Core ML export, telemetry, ANE placement problems | [README/coreml-telemetry-issues.md](../../coreml-telemetry-issues.md) |
| Anything else | `README/Notes/<topic>.md` |

`README/training-notes.md` is read at runtime by
[scripts/report_phase3.sh](../../../scripts/report_phase3.sh). Append to it; do
not move or rename it.

## When to update vs create

| Situation | Action |
| --- | --- |
| Same subsystem, new bug or follow-up | Add a section to the existing domain note |
| New symptom, same root area | Same file, new issue block using [Notes-template](../../Templates/Notes-template.md) |
| Entirely new domain with no home | Create `README/Notes/<topic>.md` |

## Consolidation checklist

1. **Search** `README/Notes/` and the two legacy notes above for the subsystem
   or keywords before writing.
2. **One entry point per domain** — one file for the export pipeline, not one
   file per incident date.
3. **Active issues first** — keep current problems near the top; move resolved
   items down or mark resolved clearly.
4. **Link out** to guides for long explanations; keep the note to summary,
   symptom, fix, verification.

## Anti-patterns

- A new markdown file for every investigation when an existing note fits.
- Duplicating the same procedure in three notes — link to the canonical guide
  once.
- Notes that belong in `README/Guides/` or `docs/` — move durable how-tos to a
  guide; keep notes for time-bound debugging trails.

## Related

- [Notes template](../../Templates/Notes-template.md)
- `write-notes` skill (`.claude/skills/write-notes/`)
