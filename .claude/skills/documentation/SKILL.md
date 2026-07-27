---
name: documentation
description: Write or review inline code documentation that captures domain knowledge, non-obvious constraints, non-greppable cross-file contracts, and state lifecycle. Use when the task is adding or reviewing Python docstrings, file headers, state docs, or constant rationale. Do not use for README files, markdown docs, plan documents, or code changes where comments are incidental.
---

# Documentation

## Purpose

Enforce the repo's inline code documentation standards. Document what a future
editor cannot safely derive from the code itself.

## Use When

- Adding or reviewing Python docstrings or file headers.
- Documenting state lifecycle, constant rationale, or MPS / Core ML export
  gotchas.
- A review flags missing or low-quality documentation.

## Do Not Use When

- Writing README, plan, or notes documents (use `markdown` / `write-notes`).
- General code changes where documentation isn't the focus.
- Markdown formatting issues (that's linting, not documentation).

## Procedure

1. Read [references/index.md](references/index.md) first.
2. Inspect the target file and the smallest set of related files needed to
   confirm what context is truly missing.
3. Add or tighten docs only where they capture:
   - domain knowledge — SSM recurrence, CTC vs RNN-T contracts, subsampling
     factors, streaming chunk semantics
   - non-obvious constraints — MPS op gaps, dtype and device rules, static
     shapes required for export, sequence-length ceilings
   - non-greppable cross-file contracts — "must match the CTC head in
     `train.py`", "stride must match `scripts/export_coreml.py`", "vocab size
     must match `utils/tokenizer.py`"
   - state lifecycle or constant rationale — SSM hidden state, streaming caches,
     what survives a checkpoint; why `d_model=256`, why `state_dim=16`
4. Prefer short, durable comments over boilerplate:
   - short file headers
   - docstrings that explain **why** or **constraints**
   - state docs that explain lifetime and persistence
   - constant comments that explain why the value exists
5. Do not add manual call graphs, line-by-line prose, or comments that are more
   likely to drift than to help.
6. If the missing context actually belongs in a canonical guide, update the
   guide as well instead of burying the whole explanation in code comments.

## Follow the existing header pattern

Several modules already carry a "Related documentation" block pointing at
`README/Mamba-on-Apple-Silicon.md` **with a section number** — see
[utils/metrics.py:47](../../../utils/metrics.py),
[utils/tokenizer.py:54](../../../utils/tokenizer.py),
[benchmarks/bench_mps.py:41](../../../benchmarks/bench_mps.py),
[train_CTC.py:52](../../../train_CTC.py). Keep that pattern: a one-line pointer
in code, the long explanation in the guide.

Those citations are **load-bearing** — if you renumber sections in that guide,
the pointers go stale silently. Grep before renumbering:

```bash
grep -rn "Mamba-on-Apple-Silicon" --include="*.py" .
```

## References

Read [references/index.md](references/index.md) first.

## Handoff Rules

- Hand off to **`debug`** if the real issue is a runtime bug, not missing docs.
- Hand off to **`markdown`** if the work is markdown documents, not inline docs.
- Hand off to normal refactoring flow if the task is structural change, not
  documentation.
