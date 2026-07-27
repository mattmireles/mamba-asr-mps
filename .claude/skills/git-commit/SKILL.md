---
name: git-commit
description: Produces comprehensive Git commit messages (what and why) and, when authorized, stages and commits. By default stages the entire working tree (all pending changes), including work from other agents or earlier sessions—not only files the current agent touched. After a successful commit (non-blocking), runs the CTC sanity train at the repo root and a **lightweight** read of **HEAD** only (bug risks, god-module smells, needless complexity, duplication)—not a full **audit** (no grades, no multi-agent); for that, the user invokes the **`audit`** skill. Surfaces findings as optional heads-up, never as a reason to block or unwind the commit unless the user asks. No automatic fixes. Message-only requests use the same template without running git until commit is explicitly requested. Also use inside authorized workflows (for example execute-plan phase commits). Do not use for read-only review, hypothetical history, or when git writes are not authorized.
---

# Git Commit

## Purpose

Produce **reviewable commits**: a clear subject line and a body that explains
**what** changed and **why**. **Staging is whole-tree by default** so one commit
reflects everything currently pending on the branch, not a cherry-pick of what
this session edited.

## Authority

- **`git commit` allowed when** the user explicitly asks you to commit, or a
  checked-in workflow skill (for example `execute-plan`) authorizes commits as
  part of its procedure. [CLAUDE.md](../../../CLAUDE.md) is explicit: *only
  commit to Git when asked.*
- **Message text only** when the user wants a commit message but did **not** ask
  you to run Git: use the subject and body rules below, optionally after
  `git status` / `git diff` for context. Do **not** `git add` or `git commit`.
- If committing would be a surprise — no explicit commit request and no workflow
  granting commit authority — **stop** before `git commit` and confirm intent.

## Use When

- A commit with a strong message is requested or workflow-authorized.
- The user wants help composing commit message text (with or without you running
  `git commit` — see Authority).

## Do Not Use When

- Git writes are forbidden by context and nothing overrides that.

## Staging scope

- **Default (this skill alone):** Stage **all** changes in the working tree that
  Git will track — modified, deleted, and **untracked** files — typically
  `git add -A` at the repo root. That includes edits made by **other agents**,
  the user, or tooling. `.gitignore` still applies (do not fight it to force
  secrets or artifacts into the commit).
- **Parent workflow override:** If the active procedure **explicitly** limits
  what may be staged (for example `execute-plan` phase commits: only files for
  the completed phase), **follow that staging scope** and still use the message
  rules below.
- If `git status` shows something surprising (unexpected paths, credentials,
  huge artifacts), **surface it** before committing; do not silently drop
  co-workers' changes to keep the commit "tidy."

### Repo-specific staging hazards

Check for these before `git add -A`:

| Hazard | What to do |
| --- | --- |
| Bulk `swift/MambaASRRunner/.build/**` deletions | These are tracked build artifacts being removed to match `.gitignore`'s `swift/**/.build/` rule. Committing them is correct, but **say so in the body** — a reviewer seeing 150+ deleted files deserves the one-line explanation. |
| `checkpoints/`, `*.pt`, `*.pth`, `*.ckpt`, `*.mlpackage/` | Already gitignored. If one appears as untracked-and-stageable, something forced it — stop and ask. |
| `.env`, `*.key`, `*.pem`, `credentials.json` | Gitignored. If any of these reach the staged set, **abort the commit** and tell the user. |
| Large `.npy` / `.wav` / CSV outputs under `benchmarks/` or `scripts/` output dirs | Confirm intent before committing measurement artifacts. |

## Procedure

1. **Inspect** full `git status` and `git diff` (and `git diff --staged` if
   anything is already staged). Understand **everything** that will be included.
2. **Stage** per [Staging scope](#staging-scope) above.
3. **Subject line** (about 50 characters target, 72 hard cap):
   - Imperative mood: "Add", "Fix", "Refactor" — not past tense ("Added",
     "Fixed") or third-person singular ("Adds", "Fixes") as the **subject**.
   - The **body** may include issue closers such as `Fixes #123` when that is
     the project convention.
   - Be specific about area or behavior, not "Update code" or "WIP".
   - The repo's existing history uses conventional prefixes
     (`fix:`, `chore:`, `docs:`, `ci:`) — match it.
4. **Body** (one blank line after the subject; wrap near 72 columns):
   - **What**: bullets or short paragraphs covering **all** substantive areas in
     this commit (not only the files you touched this turn).
   - **Why**: motivation, tradeoffs, what was broken or awkward, or why this
     approach over alternatives.
   - **Context** (optional): issue links, plan paths, or follow-ups when they
     help the next reader.
5. **Commit** (commit path only), for example in bash or zsh:
   - `git commit -m "subject" -m $'paragraph...\n\n- bullet'`
   - or `git commit` with an editor when the body is long.
6. **Verify** (commit path only): `git show --stat HEAD` matches intent (entire
   staged set landed).
7. **Post-commit audit** (commit path only): after the commit succeeds, follow
   [Post-commit audit](#post-commit-audit). Run the **CTC sanity train** at the
   repo root; then scan **only** the `HEAD` diff for bug risks, god modules,
   needless complexity, and duplication (**lightweight** — see
   [What this is not](#what-this-is-not)). **Tell the user** about findings in
   the same turn (paths, severity, next step if obvious) — as **heads-up**, not
   scolding. The commit is already done; do **not** delay the commit, imply it
   was a mistake, or push amend/revert unless the user asks. Do **not** silently
   "fix" findings unless the user asked; surfacing is the goal. If everything
   passes and nothing worrisome stands out in the diff, you may omit commentary.

On the **message-only** path, perform step 1 as needed for context, skip staging
and steps 5–7, and output the subject and body (use the template below).

## Message template

```text
fix/add/refactor: <behavior> in <area>

What:
- ...

Why:
- ...
```

## Post-commit audit

Run this **only after** `git commit` completes successfully (same repo, same
branch).

### Intent (non-blocking)

- **Commit always wins.** Nothing here vetoes, reorders, or shames a commit that
  already landed. People should keep committing; this step is **extra
  awareness** for the author, not policy for the team.
- **Alerts, not gates.** Findings are **heads-up** so you can fix forward (the
  sanity train failed after commit — here is the output) or choose to ignore.
- **Optional follow-up.** Suggest a fix commit or local cleanup only when
  helpful; never pressure amend/revert unless the user explicitly wants that.
- **Deeper review:** For **A–F grades**, optional **multi-agent** passes, and
  broader scope, the user should invoke the **`audit`** skill (message contains
  **`audit`**) — do not inflate post-commit into a full audit.

### What this is not

Stay **narrow** so commits stay fast:

- **Do** run the **CTC sanity train** and read **`git show -p HEAD`** (this
  commit only).
- **Do not** run full export suites, benchmark sweeps, or LibriSpeech training
  runs here by default — those belong to **`audit`**, **`bakeoff`**,
  **`deploy`**, or the user's task unless already requested.

### The mechanical check

This repo has **no test suite**. Never write "pytest: pass." From the
**repository root**:

```bash
PYTHONPATH="$PWD" PYTORCH_ENABLE_MPS_FALLBACK=1 python3 train_CTC.py --epochs 1 --sanity
```

This is exactly what CI runs. If the diff touched the RNN-T path, add
`python3 train_RNNT.py --epochs 1 --sanity`. If it touched Core ML export, add
`python3 scripts/export_and_validate.py`.

If the run fails, **report the relevant output** with neutral framing
("Heads-up: the CTC sanity train failed after your commit"). The commit stays;
the user can fix in a follow-up. Do not hide failures.

If `test_*.py` files exist by then, also run `python3 -m pytest`.

### Diff scope

Use the committed change as the source of truth — e.g. `git show HEAD` or
`git show -p HEAD`, and the paths touched:
`git diff-tree --no-commit-id --name-only -r HEAD`.

### Bug scan

Re-read the change as a quick self-review **on changed lines only**. Flag
**likely** problems for the user (this is not a substitute for CI or
**`audit`**):

- Logic errors, wrong conditions, off-by-one, incorrect defaults.
- **Shape and length errors** — CTC output lengths after subsampling, RNN-T
  joint dimensions, mask/padding misalignment.
- **Device and dtype** — a tensor that silently lands on CPU, an fp16/fp32
  boundary crossing, a `.to(device)` that got dropped.
- **MPS hazards** — an op with no MPS kernel newly placed on the hot path
  (works only because `PYTORCH_ENABLE_MPS_FALLBACK=1` silently moves it to CPU).
- **SSM numerical stability** — discretization, cumulative products, anything
  that can produce `NaN`/`inf` over long sequences.
- **Export traceability** — data-dependent control flow or dynamic shapes newly
  introduced on a path that `scripts/export_coreml.py` traces.
- **CI breakage** — a new top-level import of `coremltools`, `librosa`,
  `soundfile`, or `torchaudio` on the `train_CTC.py --sanity` path. CI installs
  only `requirements-ci.txt` (torch + numpy) on CPU-only Linux. This is the most
  common way a locally-green change turns CI red.
- Missing or broken error handling / early returns where failures are plausible.
- Obvious regressions: removed guards, weakened validation.
- API or type mismatches, impossible states, or changes that contradict the
  commit message.
- **Security / privacy (glance):** obvious secrets or paths logged in the diff;
  unvalidated `torch.load` on a user-supplied checkpoint path.
- Anything that would make you say "wait, that can't be right" on a PR.

### God module scan

Align with repo philosophy: **clear separation of concerns**, no dumping
unrelated responsibilities into one place. [CLAUDE.md](../../../CLAUDE.md) aims
for **under 1k LOC per file**.

Three files already blow past that — `train_RNNT.py`, `train.py`, and
`train_CTC.py`. Flag when this commit **grows** any of them, or creates a new
grab-bag module:

- **Size**: a file approaching or exceeding ~1000 LOC.
- **Scope creep**: unrelated domains or layers fused into a single module —
  data loading + model construction + training loop + export in one file.
- **Fan-in smell**: a change that makes one file the obvious choke point for
  unrelated call sites when a split would be natural.

When in doubt, **flag lightly** with reasoning; avoid crying wolf, but do not
skip obvious smells to avoid bothering the user.

### Needless complexity and duplication

- **Needless complexity**: extra layers, over-abstraction, clever patterns where
  a straight line would do, new dependencies for trivial wins, configuration
  explosions across `config/` and `hparams/`, or branching that obscures actual
  behavior — especially when it violates **simpler is better** for this repo.
- **Duplication**: this repo already carries known parallel implementations —
  `train.py` / `train_CTC.py` / `train_RNNT.py`, and `modules/rnnt_loss.py` /
  `modules/rnnt_loss_mps.py`. Flag when a commit **adds a fourth copy** of an
  existing rule, or changes one side of an existing pair without the other.

Flag when this commit **introduces or worsens** these; cite paths and why it
matters.

### What to output

- **Sanity train**: pass/fail; if fail, enough output to act on, framed as
  post-commit awareness — not a blocked workflow.
- **If other issues**: short summary, bullet list with **file paths**, what you
  saw, and whether it looks like a definite bug vs. a risk vs. a
  maintainability smell.
- **If clean**: say nothing or one line — no boilerplate required.

## Anti-patterns

- One-word subjects ("fix", "wip", "updates").
- A body that only repeats the subject.
- **Cherry-staging** only the files you personally edited when the user asked
  for a commit and no narrower workflow override applies — other agents' work
  must ship too.
- Claiming a "single logical change" while omitting co-present dirty files
  without explicit user direction.
- **Skipping the post-commit check** after a successful commit when this skill's
  commit path ran — or staying silent when it fails.
- **Reporting "pytest: pass"** — there are no tests in this repo.
- **Turning post-commit into a full audit:** multi-agent delegation or **A–F
  grades** here — use the **`audit`** skill when the user says **`audit`**.
- **Blocking mindset:** implying the user should not have committed, or urging
  amend/revert, because of post-commit findings — unless they asked for that.

## Related skills

- **`audit`:** user message includes **`audit`**; broader scope, **A–F** grades,
  optional multi-agent.
- **`git-push`:** commit, merge, push, chase CI until green.
- **`deploy`:** what "ship" means for this repo.
