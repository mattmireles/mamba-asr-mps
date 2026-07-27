# Notes

Time-bound institutional memory: debugging trails, investigation logs, run
results, and audit findings. Durable how-tos belong in
[README/Guides](../Guides) or [docs/](../../docs) instead.

Written and consolidated with the `write-notes` skill
(`.claude/skills/write-notes/`) plus the
[Notes template](../Templates/Notes-template.md). Read the
[notes consolidation guide](../Guides/content/notes-consolidation-guide.md)
before adding a file — prefer appending to an existing domain note.

## Existing notes

| Note | Covers |
| --- | --- |
| [README/training-notes.md](../training-notes.md) | Training runs, loss curves, benchmark findings, MPS behavior |
| [README/coreml-telemetry-issues.md](../coreml-telemetry-issues.md) | Core ML export and ANE placement problems |

Both live at the top level of `README/` rather than in this directory.
`README/training-notes.md` is read at runtime by
[scripts/report_phase3.sh](../../scripts/report_phase3.sh) — append to it, do
not move it.

New notes with no existing home go in this directory, named by topic
(`selective-scan-mps.md`, not `2026-07-26-debugging.md`).
