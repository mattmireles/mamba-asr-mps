# Plans

Implementation plans, scoped into phases with per-phase verification. Written
with the `create-plan` skill (`.claude/skills/create-plan/`) using the
[Plans template](../Templates/Plans-template.md), executed with `execute-plan`,
and reviewed per phase with `phase-audit`.

The workflow contract is
[README/Skills/plan-workflow-skills-guide.md](../Skills/plan-workflow-skills-guide.md).
The review checklist is
[README/Skills/phase-audit-rubric.md](../Skills/phase-audit-rubric.md).

## Existing plans

| Plan | Covers |
| --- | --- |
| [v1-ship-ctc29.md](v1-ship-ctc29.md) | **Active.** v1 ship: CTC-29 end-to-end — env+data → contracts+parity → train → export/eval/handoff |
| [README/implementation-plan.md](../implementation-plan.md) | Original Phase 1–4 roadmap (historical) |
| [README/implementation-plan-v2.md](../implementation-plan-v2.md) | Predecessor roadmap; Phase 3 optimization (superseded by the audit — see `CLAUDE.md` Ground truth) |

Both live at the top level of `README/`. `implementation-plan-v2.md` is read at
runtime by [scripts/report_phase3.sh](../../scripts/report_phase3.sh) and
[scripts/run_phase2_baselines.sh](../../scripts/run_phase2_baselines.sh) — do
not move it.

New plans go in this directory.

## Verification is not optional

Every phase needs a verification command with an observable result. This repo
has no test suite; the gate is:

```bash
PYTHONPATH="$PWD" PYTORCH_ENABLE_MPS_FALLBACK=1 python3 train_CTC.py --epochs 1 --sanity
```

See [The Mechanical Gate](../Skills/plan-workflow-skills-guide.md#the-mechanical-gate-repo-specific)
for the per-surface checks.
