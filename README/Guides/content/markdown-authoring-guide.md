# Markdown authoring guide

## Purpose

Consistent, maintainable markdown across `README/`, `docs/`, guides, plans, and
notes.

## Core rules

- Use **real markdown links** `[text](path)` for internal paths; avoid bare URLs
  in prose when a label helps.
- **Blank lines** around headings, lists, and fenced blocks so diffs and
  renderers stay predictable.
- **Language-tag** fenced code blocks when the language is known
  (` ```python `, ` ```bash `, ` ```swift `).
- **One trailing newline** at end of file.
- Prefer **one canonical explanation** — link to it instead of copying
  paragraphs across files.

## Document families

| Family | Template | Location |
| --- | --- | --- |
| Plan | [Plans-template](../../Templates/Plans-template.md) | `README/Plans/...` |
| Note | [Notes-template](../../Templates/Notes-template.md) | `README/Notes/...` |
| Guide | [guide-template](../../Templates/guide-template.md) | `README/Guides/...` |
| ADR | [ADR-template](../../Templates/ADR-template.md) | `README/Plans/...` |

Preserve each document's existing structure unless the task is an explicit
restructure.

## Legacy paths — link, do not move

Three top-level `README/` files are load-bearing and referenced from **code and
shell scripts**, not just prose:

| File | Referenced by |
| --- | --- |
| `README/Mamba-on-Apple-Silicon.md` | `train_CTC.py`, `train_RNNT.py`, `utils/metrics.py`, `utils/tokenizer.py`, `benchmarks/bench_mps.py` |
| `README/training-notes.md` | `scripts/report_phase3.sh` (reads at runtime) |
| `README/implementation-plan-v2.md` | `scripts/report_phase3.sh`, `scripts/run_phase2_baselines.sh` (read at runtime) |

Link to them from `README/Guides/`, `README/Notes/`, and `README/Plans/`. Do not
relocate them without updating every reference above.

## Plans and notes

- Plans: phases, checkboxes, and links to shared contracts stay in sync with
  implementation.
- Notes: use the issue template sections; keep investigation logs dated and
  short.

## Verification

This repo has **no configured markdown lint command**. Verify by reading the
rendered diff and checking that every internal link resolves:

```bash
grep -oE '\]\([^)#][^)]*\.md[^)]*\)' README/Guides/content/markdown-authoring-guide.md
```

## Related

- `markdown` skill (`.claude/skills/markdown/`)
- [Notes consolidation guide](notes-consolidation-guide.md)
