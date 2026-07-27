---
name: audit
description: Triggered when the user's message includes the word **audit** (primary routing hook). Findings-first review of the mamba-asr-mps repo or a scoped slice (paths, diff, commits)—runs the CTC sanity train (and surface-specific checks) as mechanical signals, optionally delegates readonly subagents by charter when scope or risk warrants it (**when in doubt, parallelize**), merges and dedupes findings, and assigns A–F grades for architecture, correctness risk, and complexity debt. Do not use when the user wants implementation fixes unless they explicitly ask to fix issues after the audit—for plan-phase checklists against an active plan, prefer phase-audit.
---

# Audit

## Purpose

Run a **structured, paranoid-friendly** audit focused on **bugs** and **needless
complexity** in this **Mamba ASR on Apple Silicon** repo — MPS training, SSM
kernels, CTC and RNN-T paths, Core ML export, and the Swift runner. The
orchestrator **does not** silently rewrite code; it **surfaces** issues with
severity, paths, and letter grades.

**Posture:** use **judgment**. Narrow scope (few files, small diff, localized
change) can be a **single-agent** pass. **Whole-repo**, **large diff**, **high
blast radius**, or **you are unsure** → prefer **multiple readonly subagents in
parallel** (one turn, several delegated calls), each with a **narrow charter**,
then **merge and dedupe** into one report. **When in doubt, use multiple
agents.**

## Use When

- The user's message includes **`audit`** as a substring (primary trigger) —
  e.g. **audit**, **audit this**, **security audit**, **use the audit skill**
  (case insensitive in normal routing).

## Do Not Use When

- The user wants **implementation** only — unless they ask for fixes **after**
  the audit lands.
- The work is **only** validating an `execute-plan` phase against the plan and
  rubric — use **`phase-audit`** instead (this skill is broader).

## Scope (ask or infer once)

Establish **audit scope** before delegating:

| Kind | How to bound |
| --- | --- |
| **Whole codebase** | Repo root; subagents apportion by area (`modules/`, `train*.py`, `scripts/`, `utils/`, `config/`, `datasets/`, `swift/`) or by charter only. |
| **Paths** | User-provided globs or directories; all charters focus there. |
| **Git delta** | User provides base (`main`, `origin/main`, tag) or "last N commits"; use `git diff`, `git log -n`, `git diff-tree --name-only`. |
| **Single feature** | User names a flow (e.g. "selective scan MPS kernel", "RNN-T loss", "Core ML export"); map to directories from `README.md` and `docs/SYSTEM_ARCHITECTURE.md`. |

State the **chosen scope** in the final report header.

## Mechanical signals (orchestrator, once)

**This repo has no test suite.** There are no `test_*.py` files and no
`pyproject.toml`. Do not report "pytest: pass" — report what you actually ran.

From **repository root**:

1. **CTC sanity train** — the primary gate, identical to CI:

   ```bash
   PYTHONPATH="$PWD" PYTORCH_ENABLE_MPS_FALLBACK=1 python3 train_CTC.py --epochs 1 --sanity
   ```

2. **Surface-specific checks** — run the ones the audited scope touches:

   | Surface touched | Check |
   | --- | --- |
   | RNN-T path | `PYTHONPATH="$PWD" python3 train_RNNT.py --epochs 1 --sanity` |
   | Core ML export | `python3 scripts/export_and_validate.py` |
   | SSM kernels | `python3 benchmarks/bench_selective_scan.py` |
   | MPS performance | `python3 benchmarks/bench_mps.py` |
   | Swift runner | `swift build -c release --package-path swift/MambaASRRunner` |

3. **pytest** — only if the scope added `test_*.py` files. Otherwise note
   "no tests in repo."

4. **Lint** — none configured (no `pyproject.toml`, no ruff/flake8 config).
   Note "no configured lint"; do not invent one.

5. **Import smoke** — cheap and catches the most common CI break:

   ```bash
   python3 -c "import ast,sys;[ast.parse(open(f).read(),f) for f in sys.argv[1:]]" $(git ls-files '*.py')
   ```

Report failures as **P0 / Critical** findings with command output (summarize if
huge). Do not "fix" unless the user later asks.

When subagents run, they should assume mechanical checks ran **once**; they
focus on **review**, not on re-running gates unless a charter requires
spot-checking a file.

## Subagent delegation (use judgment)

**When one agent is enough:** tiny or localized scope (a handful of files, one
module, or a single focused diff); low coupling; user asked for a quick pass.
The orchestrator performs the full charter coverage **solo** (still hit
architecture, correctness, ops, and complexity — just in one pass).

**When to parallelize:** whole-repo or multi-surface audit; large `git` range;
numerically sensitive paths (SSM recurrence, loss functions); export or
device-placement concerns; or **any uncertainty** about depth. **When in doubt,
launch parallel subagents.**

**If parallelizing:** in **one assistant turn**, launch **up to four** readonly
subagents. Give each one:

- The **exact scope** (paths, diff summary, or "whole repo").
- One **charter** from below (copy the charter text into the task prompt).
- Instruction: **concrete file paths**, **severity** (P0–P3), **one paragraph
  max per finding**, **no generic advice**.

### Charter 1 — Architecture & modules

God modules; separation between **model code** (`modules/`), **training
orchestration** (`train*.py`), **tooling** (`scripts/`), and **runtime-facing**
Swift/Core ML artifacts; public surfaces; fan-in choke points; misplaced
orchestration vs model math.

**Repo norm is well under 1k LOC per file.** `train_RNNT.py` (~54k bytes),
`train.py` (~52k), and `train_CTC.py` (~22k) already violate it — flag growth in
those files as a finding and note the three-way duplication between the training
entry points.

### Charter 2 — Correctness & reliability

Logic bugs; **tensor shape and length mistakes** (CTC output lengths after
subsampling, RNN-T joint dims, mask/padding alignment); **SSM numerical
stability** (discretization, cumulative products, `NaN`/`inf` over long
sequences); **device and dtype mismatches** (silent CPU landings, fp16/fp32
boundary crossings); **MPS op gaps** and whether
`PYTORCH_ENABLE_MPS_FALLBACK=1` has quietly become the hot path; tracing/export
mismatches and data-dependent control flow that breaks `coremltools.convert`;
error paths and swallowed exceptions; boundary conditions for streaming chunk
sizes and Swift-side alignment.

### Charter 3 — Security, privacy, operational

Secrets in logs or committed credentials; unsafe subprocess usage in
`scripts/*.sh` and `scripts/*.py`; **checkpoint and dataset path foot-guns**
(unvalidated `torch.load`, absolute paths baked into config); operational
mismatches between **documented I/O** and code; **CI breakage risk** — CI is
`ubuntu-latest`, CPU only, with only `requirements-ci.txt` (torch + numpy)
installed, so any new top-level import of `coremltools`, `librosa`,
`soundfile`, or `torchaudio` on the `train_CTC.py --sanity` path is a **P0**.

### Charter 4 — Complexity, duplication, maintainability

Needless abstraction; configuration or branching explosion across
`config/`, `hparams/{CTC,RNNT,S2S}`; **duplication with drift** between
`train.py`, `train_CTC.py`, and `train_RNNT.py`, and between
`modules/rnnt_loss.py` and `modules/rnnt_loss_mps.py`; dead code; **comment and
doc lies** vs `README/`, `docs/`, or `CLAUDE.md`; **absence of any test suite**
as a standing maintainability finding; naming that hides behavior.

## Consolidation (orchestrator)

After **subagents return** or after a **solo** review pass:

1. **Dedupe:** merge findings that cite the same root cause or file+theme.
2. **Severity:** **P0** critical (wrongness, security, data loss, CI/build
   breaks), **P1** high, **P2** medium, **P3** low / hygiene.
3. **Grades:** assign three letter grades **A–F** using
   [Grading rubric](#grading-rubric). **Overall grade = worst of the three** (if
   any dimension is **D**, overall cannot be **B** or higher — cap at **D**
   unless you justify an exception in one sentence).
4. **New teammate test:** one line — would a newcomer likely break this area in
   week one? (y/n + why)

## Grading rubric

Assign **Architecture**, **Correctness risk**, and **Complexity debt**
separately.

| Grade | Meaning |
| --- | --- |
| **A** | Clear boundaries, hard to misuse, simple where it matters, issues are cosmetic. |
| **B** | Solid with minor debt; a few focused fixes would raise to A. |
| **C** | Meaningful issues; would block merges in a strict shop without a plan. |
| **D** | Serious structural or reliability risk; bounded scope but unsafe. |
| **F** | Unsafe, incomprehensible, or broken; needs pause and redesign or revert. |

**Correctness risk** includes likelihood of **latent bugs** and **failure-mode**
holes — not only observed failures. In a repo with **no automated tests**,
correctness risk starts elevated and must be argued down with evidence, not
assumed.

## Output template

```text
## Audit scope
- ...

## Mechanical checks
- CTC sanity train: pass | fail (summary)
- Surface checks run: ... (or "none applicable")
- pytest: no tests in repo | pass | fail
- lint: no configured lint

## Grades
- Architecture: ?
- Correctness risk: ?
- Complexity debt: ?
- Overall (worst-of-three): ?

## Findings (severity order)
### P0 — Critical
- ...

### P1 — High
- ...

### P2 — Medium
- ...

### P3 — Low
- ...

## Delegation / overlap notes
- Solo vs N subagents; deduped: ...

## Residual risks / what we did not run
- ...

## New teammate test
- ...
```

## Anti-patterns

- **Whole-repo or high-risk audit in a single shallow pass** when depth was
  needed — **when in doubt, parallelize.**
- **Four subagents for a three-line diff** — wasted latency; judge
  proportionally.
- **Reporting "pytest: pass"** — there are no tests. Report what ran.
- **Findings without paths** — every substantive issue should anchor to a file
  or symbol when possible.
- **Auto-implementing** during audit when the user only asked for review.
- **Grade inflation** — if Correctness is **D**, overall is not **B**.
- **Ignoring mechanical failures** — a red sanity train is at least **P0**.

## Relation to other skills

- **`phase-audit`:** plan-phase completion vs rubric and plan checkboxes.
- **`git-commit`:** post-commit **non-blocking** sanity-train + **`HEAD`** diff
  scan only — **not** a substitute for **`audit`** (no full matrix by default,
  no grades, no multi-agent).
- **`audit-fix-loop`:** this procedure plus mandatory fix iterations to **A/A/A**
  and a final commit.
- **`deploy`:** defines what "ship" means here; audit does not replace
  user-intent release checks.
