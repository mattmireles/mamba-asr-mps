---
name: markdown
description: Write or repair repo markdown files. Use when the task is editing README files, guides, plans, notes, or markdown lint failures. Apply the repo's markdown rules, preserve the right template or document structure, and keep prose tight. Do not use for inline code comments or code changes that only happen to touch markdown strings.
---

# Markdown

## Purpose

Use this skill to keep repo markdown clean, consistent, and easy to maintain.

## Use When

- Writing or revising markdown docs under `README/` or `docs/`.
- Fixing markdown lint failures.
- Editing plans, notes, guides, or README files.

## Do Not Use When

- The real task is inline code documentation (use `documentation`).
- The work is code implementation, not doc authoring.
- The file is not markdown.

## Procedure

1. Read [references/index.md](references/index.md) first.
2. Identify the document family:
   - guide → `README/Guides/` or `docs/`
   - note → `README/Notes/` (or the two legacy top-level notes)
   - plan → `README/Plans/` (or the two legacy top-level roadmaps)
   - general README or content doc
3. Use the canonical template when the document family has one
   ([Templates/](../../../README/Templates)).
4. Preserve the existing structure of the doc family unless the task is
   explicitly a reorganization.
5. Apply the repo markdown rules:
   - real markdown links, not bare URLs
   - no unnecessary inline HTML
   - blank lines around headings and lists
   - language-tagged fenced code blocks when known (` ```python `, ` ```bash `,
     ` ```swift `)
   - single trailing newline
6. Keep prose lean. Prefer one canonical explanation over duplicated text across
   several files.
7. Verify internal links resolve before stopping.

## Load-bearing paths — do not move

Three top-level `README/` files are referenced from **code and shell scripts**,
not just prose. Renaming or relocating them breaks things at runtime:

| File | Referenced by |
| --- | --- |
| `README/Mamba-on-Apple-Silicon.md` | `train_CTC.py`, `train_RNNT.py`, `utils/metrics.py`, `utils/tokenizer.py`, `benchmarks/bench_mps.py` — **by section number** |
| `README/training-notes.md` | `scripts/report_phase3.sh` reads it at runtime |
| `README/implementation-plan-v2.md` | `scripts/report_phase3.sh`, `scripts/run_phase2_baselines.sh` read it at runtime |

Link to them. Do not move them. Do not renumber sections in
`Mamba-on-Apple-Silicon.md` without grepping for citations first:

```bash
grep -rn "Mamba-on-Apple-Silicon\|training-notes\|implementation-plan-v2" --include="*.py" --include="*.sh" .
```

## Verification

There is **no configured markdown lint command** in this repo (no
`package.json`, no lint config). Do not claim a lint pass. Verify by checking
that every relative link resolves:

```bash
for f in $(git ls-files '*.md'); do
  grep -oE '\]\(([^)#:]+\.md)' "$f" | sed 's/](//' | while read -r l; do
    [ -e "$(dirname "$f")/$l" ] || echo "BROKEN: $f -> $l"
  done
done
```

## References

Read [references/index.md](references/index.md) first.

## Handoff Rules

- Hand off to `documentation` if the real work is inline code docs rather than
  markdown documents.
- Hand off to `write-notes` if the main question is where a note belongs and how
  to consolidate it without note sprawl.
- Hand off to `guide-ingest` if the task is importing external research into a
  new guide.

## Prose Wrapping

- Do not hard-wrap ordinary Markdown prose at a fixed column. Keep each paragraph on one source line unless Markdown semantics or a document format requires breaks (for example, lists, tables, code, blockquotes, or fixed-width email/plain-text output).
- Treat editor and browser word wrap as presentation, not a reason to insert newlines.
- Do not reflow prose merely to satisfy Markdownlint MD013. Disable or configure that rule for prose-heavy documentation when appropriate.
- Preserve existing paragraph line structure when editing unrelated text; avoid drive-by reflow.
