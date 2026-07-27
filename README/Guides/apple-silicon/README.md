# Apple Silicon Guides

MPS, Core ML, and ANE field guides. Externally researched reference material
lands here via the `guide-ingest` skill (`.claude/skills/guide-ingest/`).

Currently empty — the Apple Silicon material lives at these paths instead:

| Guide | Covers |
| --- | --- |
| [README/Mamba-on-Apple-Silicon.md](../../Mamba-on-Apple-Silicon.md) | MPS/ANE optimization strategy. Cited **by section number** from `train_CTC.py`, `train_RNNT.py`, `utils/metrics.py`, `utils/tokenizer.py`, `benchmarks/bench_mps.py` — do not move or renumber. |
| [docs/Mamba-Apple-Silicon-guide.md](../../../docs/Mamba-Apple-Silicon-guide.md) | PyTorch MPS semantics and deployment. An ingested research export — note its dated reference list. |
| [README/coreml-telemetry-issues.md](../../coreml-telemetry-issues.md) | Core ML export and ANE placement problems (a note, not a guide). |

New guides on compute-unit scheduling, op compatibility, enumerated shapes, or
stateful Core ML belong here. Add a row to
[README/Guides/README.md](../README.md) when one lands.
