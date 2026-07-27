---
name: git-push
description: Commits any pending work via git-commit, syncs with origin (fetch, merge-oriented pull), pushes the current branch, watches GitHub Actions, and fixes root causes until CI is green—including coverage gates without lowering thresholds. Use when the user asks to push, sync and push, or ship the branch and chase CI. Requires explicit invocation or a parent workflow that authorizes push and CI remediation. Do not use without permission to write to git remote or when the user forbids automated CI fix loops.
---

# Git Push

## Purpose

Close the loop from **dirty working tree → clean push → green GitHub Actions**:
commit everything pending when needed, integrate upstream with **merge** (not
silent rebase unless the user prefers otherwise), push, watch workflows, and
iterate on real failures until they pass.

## Authority

- **`git push` allowed when** the user invokes `$git-push`, names `git-push`, or
  clearly asks you to sync, push, and fix CI until green — or a checked-in
  workflow authorizes the same (for example **`execute-plan`** step 5 **after
  all phases** are committed). **`execute-plan`** does **not** authorize running
  **`git-push`** after **each** phase — only the final tail.
- This authorizes **`git commit`**, **`git pull`/`git merge`** (integrating
  `origin`), **`git push`**, reading CI logs, and **code fixes** needed to get
  workflows green.
- If push or CI remediation would be a surprise, **stop** and confirm.
  [CLAUDE.md](../../../CLAUDE.md): only commit when asked.

## Use When

- The user wants the branch **on the remote** and **CI passing**.
- You need the full "commit if needed → sync → push → babysit CI" routine.

## Do Not Use When

- The user only wants a **dry run** or local-only commands.
- You are not allowed to touch **`origin`** or fix CI (no override).

## Preconditions (agent)

1. Know the **current branch** and whether it has an **upstream**
   (`git status -sb`, `git rev-parse --abbrev-ref @{u}` when set).
2. `git fetch origin` before deciding you are in sync.

## What CI actually is here

One workflow, one job — [.github/workflows/ci.yml](../../../.github/workflows/ci.yml):

| Property | Value |
| --- | --- |
| Job | `sanity-train` |
| Runner | `ubuntu-latest` — **Linux, CPU only, no MPS, no Core ML** |
| Python | 3.11 |
| Deps | `requirements-ci.txt` only — **torch + numpy** |
| Command | `python train_CTC.py --epochs 1 --sanity` |
| Triggers | push to `main`, PRs targeting `main` |

**The dominant CI failure mode:** a change works on your Mac and fails on CI
because it added a top-level import of `coremltools`, `librosa`, `soundfile`, or
`torchaudio` somewhere on the `train_CTC.py --sanity` import path. Reproduce it
locally before guessing:

```bash
python3 -c "
import sys, importlib
blocked = {'coremltools','librosa','soundfile','torchaudio'}
class Block:
    def find_module(self, name, path=None):
        return self if name.split('.')[0] in blocked else None
    def load_module(self, name):
        raise ImportError(f'{name} not available in CI')
sys.meta_path.insert(0, Block())
importlib.import_module('train_CTC')
print('CI import path OK')
"
```

The second failure mode is anything that assumes MPS or `torch.backends.mps`
without a CPU branch.

## Procedure

### 1. Uncommitted changes

- If **`git status --porcelain`** is non-empty (including untracked you intend
  to keep), run **`git-commit`** first so the working tree is committed with a
  full **what / why** message.
- **Do not** bypass `git-commit` with a lazy one-liner unless the user
  explicitly overrides.

### 2. Integrate `origin` (merge-oriented)

- **Preferred** for this skill: after `git fetch origin`, integrate with
  **merge**, not rebase — unless the user explicitly asks for rebase.
- If an upstream exists: e.g. `git pull --no-rebase` (or `git merge`
  `origin/<upstream-branch>` after fetch) so local commits combine with remote
  updates.
- If there is **no upstream** yet, a reasonable first push is
  `git push -u origin HEAD` after the branch is ready — set upstream for future
  pulls.
- **Merge conflicts**: resolve carefully; if intent is ambiguous, **stop** and
  ask rather than guessing.

### 3. Push

- `git push` to the appropriate remote (usually `origin`) and branch.
- If push is rejected because the remote advanced, **fetch**, **merge** (per
  above), resolve conflicts, then **push again** — do not force-push unless the
  user explicitly requests it and it is safe for the branch.

### 4. Monitor GitHub Actions

```bash
gh run list --branch "$(git rev-parse --abbrev-ref HEAD)" --limit 5
```

Then watch the run for the pushed commit:

```bash
gh run watch <run-id> --exit-status
```

Wait until the workflow **finishes**.

### 5. On failure: fix root cause, repeat

- Read logs (`gh run view <run-id> --log-failed`); reproduce locally when
  possible — for this repo that usually means the import-block snippet above, or
  running the sanity train with MPS disabled:

  ```bash
  PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES="" python3 train_CTC.py --epochs 1 --sanity
  ```

- Fix the **underlying issue** — not symptoms only — then **commit** (again via
  **`git-commit`** when there are new changes) and **push**, then **re-watch**
  CI.
- **Loop** until green or until you hit a blocker (permissions, flaky external,
  ambiguous product intent).

### 6. Quality gates

- If CI fails because a **required threshold** is not met: **never satisfy the
  gate by lowering the threshold or disabling the check** as a shortcut.
- Correct response: fix the code, or **strengthen verification** so the bar is
  **earned**.
- Specifically: do **not** "fix" a failing sanity train by adding `--sanity`
  escape hatches, shrinking the synthetic batch until the assertion passes, or
  removing the CI step.

## Anti-patterns

- Pushing with a dirty tree (except when every pending change is intentionally
  left out — and then say so explicitly; default is commit first).
- Force-push to shared branches without explicit approval.
- "Fixing" CI by weakening the sanity train or deleting the workflow step.
- Adding a heavy dependency to `requirements-ci.txt` to paper over an import
  that should have stayed lazy.
- Re-running CI repeatedly without changing what failed.

## Related skills

- **`git-commit`**: message shape and default **whole-tree** staging (unless a
  narrower parent workflow applies).
- **`execute-plan`**: may narrow what gets committed **per phase**; this skill
  still applies to the **push / merge / CI loop** once commits are ready.
- **`deploy`**: what "ship" means beyond a green branch.
