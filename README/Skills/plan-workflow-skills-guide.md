# Plan Workflow Skills Guide

Canonical workflow guide for the repo's plan-oriented skills:
`create-plan`, `execute-plan`, `execute-plan-hardcore`, and `phase-audit`.

## Purpose

These skills turn a repeated manual pattern into a stable workflow:

- create a real plan from repo knowledge
- execute an approved plan one phase at a time
- audit each completed phase before moving on

Canonical knowledge stays in **`README/`** and **`docs/`** (guides, plans,
notes). This file is the workflow contract; skills should wrap it, not invent a
parallel process.

## Shared Rules

- Keep each skill narrow:
  - `create-plan` writes plans
  - `execute-plan` executes plans
  - `execute-plan-hardcore` runs the same execution loop as `execute-plan`, then
    a full **`audit`** until Architecture, Correctness risk, and Complexity debt
    are all **A** (fix and repeat)
  - `phase-audit` reviews completed phases
- If a runtime cannot support delegated review cleanly, the workflow must still
  work with a **local** audit using
  [phase-audit-rubric.md](./phase-audit-rubric.md).
- Optional cross-agent review (separate Codex / Claude Code CLI threads) is
  **nice to have**, not required for this repo unless the user asks for it.

## The Mechanical Gate (repo-specific)

This repo has **no test suite** — there are no `test_*.py` files and no
`pyproject.toml`. Do not claim "tests pass"; there are none to pass.

The real gate, and exactly what
[.github/workflows/ci.yml](../../.github/workflows/ci.yml) runs, is the CTC
synthetic sanity train:

```bash
PYTHONPATH="$PWD" PYTORCH_ENABLE_MPS_FALLBACK=1 python3 train_CTC.py --epochs 1 --sanity
```

Secondary checks, run when the change touches that surface:

| Surface touched | Check |
| --- | --- |
| RNN-T path | `PYTHONPATH="$PWD" python3 train_RNNT.py --epochs 1 --sanity` |
| Core ML export | `python3 scripts/export_and_validate.py` |
| SSM kernels | `python3 benchmarks/bench_selective_scan.py` |
| MPS performance | `python3 benchmarks/bench_mps.py` |
| Swift runner | `swift build -c release --package-path swift/MambaASRRunner` |

**Lint:** none configured. Say "no configured lint" rather than inventing one.

**pytest:** installed in the environment but zero tests exist. If a phase adds
`test_*.py` files, `python3 -m pytest` becomes a real gate — say so explicitly
in the plan.

## Side-Effect Classes

- `create-plan`: repo-write, no git side effects beyond normal file edits
- `phase-audit`: read-only review
- `execute-plan`: git-write workflow that may commit, sync, push, and monitor CI
- `execute-plan-hardcore`: same git-write surface as `execute-plan`, plus
  post-push **`audit`** iterations and fixes until **A / A / A** on the audit
  rubric (see `.claude/skills/execute-plan-hardcore/SKILL.md`)

## Authority Rule

- Explicit invocation of a workflow skill authorizes the side effects documented
  for that skill (e.g. "use execute-plan", `$execute-plan`, "use
  execute-plan-hardcore", `$execute-plan-hardcore`).
- Implicit routing does **not** authorize git writes. If `execute-plan` or
  `execute-plan-hardcore` was not invoked explicitly, prepare local changes but
  stop before commit or push and say why.

This mirrors the exception clause in [CLAUDE.md](../../CLAUDE.md).

## Skill: `create-plan`

### Job

Turn a concrete request into a repo-native implementation plan using
[Templates/Plans-template.md](../Templates/Plans-template.md).

### Research Order

1. Related files under **`README/`** and **`docs/`** — start with
   [Mamba-on-Apple-Silicon.md](../Mamba-on-Apple-Silicon.md),
   [implementation-plan-v2.md](../implementation-plan-v2.md),
   [training-notes.md](../training-notes.md),
   [coreml-telemetry-issues.md](../coreml-telemetry-issues.md), and
   [docs/SYSTEM_ARCHITECTURE.md](../../docs/SYSTEM_ARCHITECTURE.md).
2. **`CLAUDE.md`** for the MPS training and PyTorch → Core ML constraints.
3. **Context7** only when the plan depends on current `coremltools`, PyTorch
   MPS, or Apple API behavior that may have changed.

### Output Contract

- Use the plans template; include phases, verification, hard requirements, and
  rollback where relevant.
- Name concrete files when the path is knowable.
- State the per-phase verification command from
  [The Mechanical Gate](#the-mechanical-gate-repo-specific) — not "run the
  tests".

## Skill: `execute-plan`

### Job

Execute an existing checked-in plan end-to-end, **one phase at a time**.

### Required Loop (per phase)

1. Read the phase and linked guides.
2. Implement only that phase's scope.
3. Audit the phase (`phase-audit` or local rubric); fix findings before
   proceeding.
4. Update plan checkboxes to match reality.
5. Commit the phase with a clear message (**narrow** staging vs default
   `git-commit` whole-tree — see the `execute-plan` skill).

After all phases: sync, push, and monitor CI (**`git-push`**) when the user wants
the branch integrated. CI here is the single `sanity-train` job in
[ci.yml](../../.github/workflows/ci.yml).

For **`execute-plan-hardcore`**, that loop is **Part A** only; **Part B** is a
mandatory full **`audit`** on the plan-execution scope, then fix and re-audit
until all three rubric grades are **A**.

### Worktree Rule

- Ignore unrelated dirty files unless they conflict with the active phase.
- Never revert unrelated user work.
- The `swift/MambaASRRunner/.build/` tree is build output that shows as deleted
  in `git status`. It is not your change — leave it alone.

## Skill: `execute-plan-hardcore`

### Job

Same as **`execute-plan`**, then require a passing full-repo-style **`audit`**
(three dimensions **A / A / A**) on the execution scope, with authorized fix
iterations until grades hold or the user must resolve a tradeoff.

### When to use

- Explicit invocation only (`execute-plan-hardcore`, `$execute-plan-hardcore`).
- User wants plan execution **and** the audit-to-**A** gate, not
  **`execute-plan`** alone.

## Skill: `phase-audit`

### Job

Review a completed phase like a skeptical senior reviewer before the next phase
or push.

### Review Style

- Findings first, severity ordered, concrete file references.
- Read [phase-audit-rubric.md](./phase-audit-rubric.md) before auditing.

## Delegated Audit Fallback

If forked/delegated review is not available, `execute-plan` must still run the
same rubric locally. **`execute-plan-hardcore` Part B** is separate: a full
**`audit`** on the execution scope (not only `phase-audit`).

## Invocation Policy

Prefer explicit `$create-plan`, `$execute-plan`, `$execute-plan-hardcore`, or
`$phase-audit` when forcing a precise handoff.
