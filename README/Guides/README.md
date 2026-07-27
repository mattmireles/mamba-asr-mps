# Guides

Durable reference material: field guides, best practices, known bugs, edge
cases, and workarounds. This is the shelf CLAUDE.md points at when a bug gets
tricky.

Notes are for time-bound investigation trails and belong in
[README/Notes](../Notes). Plans belong in [README/Plans](../Plans).

## In this directory

| Guide | Covers |
| --- | --- |
| [content/code-documentation-guide.md](content/code-documentation-guide.md) | Inline docstring and file-header standards |
| [content/markdown-authoring-guide.md](content/markdown-authoring-guide.md) | Markdown rules and document families |
| [content/notes-consolidation-guide.md](content/notes-consolidation-guide.md) | Anti-sprawl policy for `README/Notes` |

`apple-silicon/` holds MPS, Core ML, and ANE field guides. New externally
researched guides land there via the `guide-ingest` skill.

## Elsewhere in the repo

| Guide | Covers |
| --- | --- |
| [README/Mamba-on-Apple-Silicon.md](../Mamba-on-Apple-Silicon.md) | MPS/ANE optimization strategy — cited by section number from `train_CTC.py`, `train_RNNT.py`, `utils/metrics.py`, `utils/tokenizer.py`, `benchmarks/bench_mps.py` |
| [docs/Mamba-Apple-Silicon-guide.md](../../docs/Mamba-Apple-Silicon-guide.md) | Deployment guide |
| [docs/SYSTEM_ARCHITECTURE.md](../../docs/SYSTEM_ARCHITECTURE.md) | System design |
| [docs/INTEGRATION_GUIDE.md](../../docs/INTEGRATION_GUIDE.md) | iOS/macOS integration |
| [docs/ERROR_HANDLING_GUIDE.md](../../docs/ERROR_HANDLING_GUIDE.md) | Troubleshooting |
| [docs/mamba-asr-landscape.md](../../docs/mamba-asr-landscape.md) | Architecture and research context |

`README/Mamba-on-Apple-Silicon.md` is referenced by section number from code.
Do not renumber its sections or move the file without updating every citation.
