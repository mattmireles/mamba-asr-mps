# Code documentation guide

## Purpose

Standards for **inline** documentation in this repo: docstrings, file headers,
and rationale for constants and tricky tensor shapes — not README or plan files.

## Document what code cannot show

Add or tighten docs only when they capture:

- **Domain knowledge** — SSM recurrence math, CTC vs RNN-T loss contracts,
  frontend subsampling factors, streaming chunk semantics.
- **Non-obvious constraints** — MPS op gaps and `PYTORCH_ENABLE_MPS_FALLBACK`,
  fp16/fp32 boundaries, static shapes required for Core ML export, device
  placement rules, sequence-length limits.
- **Cross-file contracts** — "output length must stay aligned with the CTC head
  in `train.py`", "this stride must match `scripts/export_coreml.py`", "the
  vocab size here must match `utils/tokenizer.py`" — when grep alone is
  insufficient.
- **State lifecycle** — SSM hidden state, streaming caches, optimizer/EMA
  state, what survives a checkpoint and what does not.
- **Constant rationale** — why `d_model=256`, why `state_dim=16`, why this
  chunk size, why this 4× time reduction.

## Prefer short and durable

- Short module docstrings when the role is not obvious from the path.
- Docstrings that explain **why** or **constraints**, not a rephrasing of the
  signature.
- Avoid manual call graphs and comments that will drift on the next refactor or
  export.

## Repo conventions already in use

Several modules already carry a "Related documentation" block in the header
pointing at `README/Mamba-on-Apple-Silicon.md` with a section number (see
[utils/metrics.py](../../../utils/metrics.py),
[benchmarks/bench_mps.py](../../../benchmarks/bench_mps.py)). Keep that pattern:
one-line pointer in code, long explanation in the guide.

## When the explanation belongs elsewhere

If the real answer is a long operational procedure, put it in a guide under
`README/Guides/` or `docs/` and link from a one-line comment in code.

## Related

- `documentation` skill (`.claude/skills/documentation/`)
- [Mamba on Apple Silicon](../../Mamba-on-Apple-Silicon.md) — MPS/ANE strategy
- [Mamba Apple Silicon guide](../../../docs/Mamba-Apple-Silicon-guide.md) — deployment
