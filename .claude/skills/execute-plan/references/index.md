# Execute Plan References

**Git cadence:** phase **commits** repeat per completed phase; **`git push`** and
the **`git-push`** skill run **once after all phases** (see parent `SKILL.md`).
Do not push after each phase unless the user explicitly overrides.

Canonical docs for `execute-plan` live under **`README/Skills/`**:

- `README/Skills/plan-workflow-skills-guide.md`
  Read first for the shared workflow contract, the explicit execution loop, and
  the repo's mechanical gate (there is no test suite here).
- `README/Skills/phase-audit-rubric.md`
  Read for the canonical local-audit fallback when delegated review is not
  available.

Always inspect:

- the concrete checked-in plan file before any implementation
- every guide or note linked from the active phase
- `README/Mamba-on-Apple-Silicon.md` when the phase touches MPS performance,
  device placement, or ANE targeting
- `README/coreml-telemetry-issues.md` when the phase touches Core ML export
- the changed files and verification outputs before updating plan checkboxes or
  committing

## CI shape

One job: `sanity-train` in `.github/workflows/ci.yml`. `ubuntu-latest`, Python
3.11, CPU only, installs `requirements-ci.txt` (torch + numpy), runs
`python train_CTC.py --epochs 1 --sanity`. A phase that adds a heavy import to
that path breaks CI even when it works locally on macOS.
