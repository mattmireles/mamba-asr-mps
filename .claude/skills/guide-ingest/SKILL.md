---
name: guide-ingest
description: >-
  Ingests offline deep-research exports into README guides: normalizes messy
  exports (escapes, data-URI images, code snippets), converts to clean markdown,
  verifies library/API claims with Context7 MCP when needed, adds cross-links
  across README/Guides, README/Notes, and docs/, and updates related docs to
  link back. Use when adding or refreshing a guide from browser-downloaded
  research, Deep Research output, or similar raw material; when the user says
  guide-ingest, ingest-guide, research-to-guide, or wants corpus cross-linking
  for a new guide.
---

# Guide ingest

## Purpose

Turn raw, offline research into a **durable repo guide** that matches the
[markdown authoring guide](../../../README/Guides/content/markdown-authoring-guide.md),
reflects **current** library and platform facts where Context7 can verify them,
and sits correctly in the web of `README/Guides/`, `README/Notes/`, and `docs/`.

`README/Guides/` is the shelf for externally created reference manuals: Deep
Research or external-agent output that has already landed as raw source
material. `README/Notes/` is for this repo's own learnings, experiment logs, and
decisions. Do not invent new guide material from local analysis alone; put local
analysis in notes and link to existing guides when useful.

**Hard provenance gate:** no external source artifact means no new
`README/Guides/` file. A prompt, checkpoint, local audit, or locally-authored
synthesis belongs in `README/Notes/` until the external guide actually lands.

[docs/Mamba-Apple-Silicon-guide.md](../../../docs/Mamba-Apple-Silicon-guide.md)
is a worked example of an ingested research export in this repo — note its
numbered reference list with access dates.

## Use When

- Importing deep-research (or similar) output into `README/Guides/...`.
- The user invokes **guide-ingest**, **ingest-guide**, **research-to-guide**, or
  the same workflow in natural language.
- A new guide needs **outbound** links to notes/guides and **inbound** links
  from existing docs.
- The source is a **messy export** (Docs/PDF/chat) with escaped punctuation,
  broken image reference blocks, or placeholder formulas.

## Do Not Use When

- The task is only a small markdown typo fix (use **`markdown`**).
- The task is only where to put a note (use **`write-notes`**).
- The work is inline code documentation (use **`documentation`**).

## Prerequisites

- **Context7 MCP** available for library/tool docs when verification is in
  scope.
- Raw source: a path or pasted content from an **externally generated** research
  artifact. If no external source exists, stop and say so rather than
  manufacturing a guide.

## Where new guides go

| Topic | Destination |
| --- | --- |
| MPS, Core ML, ANE, Apple Silicon runtime | `README/Guides/apple-silicon/` |
| Documentation, markdown, notes policy | `README/Guides/content/` |
| Deployment, integration, architecture | `docs/` (matches the existing files there) |
| Anything else | `README/Guides/<topic>/` |

## Procedure

### 1. Normalize to proper markdown

1. Confirm the source is an externally generated guide or report, and record the
   source path or URL in the guide. If the content is only repo-local
   investigation, stop and write a note instead.
2. Read the **`markdown`** skill and the
   [markdown authoring guide](../../../README/Guides/content/markdown-authoring-guide.md).
3. Produce guide-shaped markdown, using
   [guide-template.md](../../../README/Templates/guide-template.md):
   - real markdown links, not bare URLs
   - blank lines around headings and lists
   - language-tagged fences (` ```python `, ` ```bash `, ` ```swift `)
   - single trailing newline; no unnecessary HTML
4. Match the tone and top-of-file blurb of sibling guides in the destination
   folder.

### 2. Mechanical cleanup (exports from Docs, PDF, chat)

Apply when the file has obvious paste/export damage:

1. Remove trailing `[imageN]: <data:image...>` reference definitions; replace
   inline `![][imageN]` with meaningful text — recover numbers and formulas from
   images when practical, or substitute concise prose.
2. Unescape systematically (multi-character sequences first): `\[ \]`, `\!=`,
   `\==`, `\-\>`, `\**`, ` \- `, `\+`, `\>=`, `\<=`, `\>`, `\<`, `\=`, `\#`,
   `\_`, `\(`, `\)`, `\.`, `` \` ``, `\!`, then remaining `\[` / `\]` for
   citations where needed.
3. **Swift:** replace `\\(` with a placeholder **before** applying `\(` → `(`,
   then restore to `\(` (string interpolation).
4. After the main pass: `-\>` → `->`, then `\-` → `-` (CLI flags, LLDB `-n`).
5. **Python indentation survives export poorly.** Re-check every fenced Python
   block actually parses:

   ```bash
   python3 - <<'EOF'
   import ast, re, sys
   src = open(sys.argv[1] if len(sys.argv)>1 else 'GUIDE.md').read()
   for i, block in enumerate(re.findall(r'```python\n(.*?)```', src, re.S)):
       try: ast.parse(block)
       except SyntaxError as e: print(f'block {i}: {e}')
   EOF
   ```

6. Fix obviously broken snippets — e.g. `os.environ = "1"` →
   `os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"` when that is what the prose
   means.
7. `grep` the file for `\\` after edits; only intentional escapes should remain.

### 3. Verify claims with Context7 (fix stale deep-research)

1. Extract **verifiable** claims: library APIs, CLI flags, framework behavior,
   config keys, deprecation notices, default versions.
2. For each claim, use Context7: resolve the library ID, then pull the relevant
   docs. Prefer **current** official behavior over the research draft.
3. **Update the guide** when Context7 contradicts the export: correct API names,
   replace deprecated patterns, fix version-specific statements.

**Highest-risk claims in this repo's domain**, in order — check these first:

- **PyTorch MPS op coverage.** Changes every minor release. A guide claiming
  "op X is unsupported on MPS" goes stale fast.
- **`coremltools` conversion API** — `convert_to`, `minimum_deployment_target`,
  `compute_precision`, `StateType` availability.
- **Core ML deployment targets** and which iOS/macOS version gates which op.
- **`torch.export` vs `torch.jit.trace`** guidance — actively shifting.

4. For claims **outside** Context7's scope (product strategy, subjective
   opinion, internal project behavior), do **not** fake verification. Leave as
   narrative, mark as opinion/heuristic, or point to a repo note that is
   authoritative.

### 4. Add outbound links (guides and notes)

1. While reading sections, identify concepts that already have a home in
   `README/Guides/`, `README/Notes/`, `docs/`, or
   `README/Mamba-on-Apple-Silicon.md`.
2. Add **inline** markdown links on the first strong mention in each section, or
   a compact "Related" list. Use **repo-relative** paths from the new guide file.
3. Prefer linking to **one canonical** guide per topic rather than duplicating
   long explanations.

### 5. Add inbound links (corpus updates)

Goal: related docs **point back** to the new guide so agents discover it from
both directions.

1. **Discover candidates**: search `README/`, `docs/`, and the guide index for
   overlapping keywords, library names, and headings.
2. **Edit sparingly**: add a link in "Related", "See also", or the most relevant
   paragraph — **minimal** diff, no drive-by rewrites.
3. Add a row to the table in
   [README/Guides/README.md](../../../README/Guides/README.md) so the new guide
   is indexed.
4. If a note covers the same subsystem, add a short bullet there with a link
   (see **`write-notes`** for consolidation habits).

### 6. Close the loop

1. Verify every internal link in the new guide resolves (see the `markdown`
   skill's verification snippet). There is **no markdown lint** configured — do
   not claim one ran.
2. If the ingest touched anything beyond markdown, run the mechanical gate:
   `PYTHONPATH="$PWD" python3 train_CTC.py --epochs 1 --sanity`.
3. Give the user a **short summary**: mechanical fixes applied, what was
   corrected via Context7, which files gained inbound links, and any claims left
   unverified.

## Handoff Rules

| Situation | Hand off to |
| --- | --- |
| Repo markdown rules, structure | **`markdown`** |
| Where a note should live, note sprawl | **`write-notes`** |
| Python docstrings in code | **`documentation`** |
| The material is a local investigation, not external research | **`write-notes`** |

## References

- [Markdown authoring guide](../../../README/Guides/content/markdown-authoring-guide.md)
- [Notes consolidation guide](../../../README/Guides/content/notes-consolidation-guide.md)
- [Guide template](../../../README/Templates/guide-template.md)
- `markdown` skill: [references/index.md](../markdown/references/index.md)
