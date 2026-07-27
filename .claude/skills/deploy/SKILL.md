---
name: deploy
description: >-
  Clarifies what "ship" means for mamba-asr-mps: there is no server or cloud
  deploy pipeline. Use for tagging releases, pushing the branch, handing Core ML
  artifacts to a consuming iOS/macOS app, or verifying exports before handoff.
  Before treating a revision as releasable, run the repo's primary checks (see
  git-commit / audit). Do not use when the user only wants local experiments or
  training runs with no remote or release intent.
---

# Deploy (mamba-asr-mps)

## Purpose

This repository is a **PyTorch/MPS training and Core ML export tree** for Mamba
ASR models. It has **no server, no Workers, no cloud deploy scripts**. Nothing
here is deployed in the web sense.

"Deploy" in this context means **release engineering** appropriate to this repo:

- **Git:** push branches, tags, or PRs (usually via **`git-push`**).
- **Artifacts:** `.mlpackage` bundles produced by
  [scripts/export_coreml.py](../../../scripts/export_coreml.py), checkpoints,
  and the tokenizer/vocab, consumed by a separate iOS/macOS app.
- **Swift runner:** `swift/MambaASRRunner` — the reference host for the exported
  model.
- **Verification:** prove the export is faithful and fast enough *before*
  handing it downstream.

## Default interpretation

**"Ship it" / "deploy"** → confirm what the user means:

| They probably mean | Route to |
| --- | --- |
| Push the branch, get CI green | **`git-push`** |
| Cut a release tag | `git tag` + push, after the pre-ship gate |
| Hand a model to the app | Export → validate → profile → package (below) |
| Just run training | Not this skill |

Do not assume a cloud deploy — there isn't one.

## Pre-ship gate

From the **repository root**. This repo has **no test suite**; these commands
are the gate.

### Always

```bash
PYTHONPATH="$PWD" PYTORCH_ENABLE_MPS_FALLBACK=1 python3 train_CTC.py --epochs 1 --sanity
```

### When the change touches the RNN-T path

```bash
PYTHONPATH="$PWD" python3 train_RNNT.py --epochs 1 --sanity
```

### When shipping a Core ML artifact

1. **Export and validate** — never hand over an unvalidated `.mlpackage`:

   ```bash
   python3 scripts/export_and_validate.py
   ```

2. **Numerical parity** against the PyTorch reference — use the
   [`coreml-validate`](../coreml-validate/SKILL.md) skill. Correlation and max
   error, not vibes.

3. **Compute placement** — use the [`coreml-profile`](../coreml-profile/SKILL.md)
   skill. Confirm the model actually lands where you claim it does. `.all` is a
   request, not a guarantee.

4. **Accuracy** — WER/CER on a real eval set, not synthetic:

   ```bash
   python3 scripts/compute_wer_cer.py
   ```

5. **Swift host builds and runs**:

   ```bash
   swift build -c release --package-path swift/MambaASRRunner
   ```

### CI

CI is one job — CPU-only Linux running the CTC sanity train. It is a **smoke
test, not a release gate.** A green CI run tells you the code imports and one
synthetic epoch completes on CPU. It tells you nothing about MPS behavior, Core
ML export fidelity, ANE placement, or WER. Do not present green CI as proof a
model is shippable.

## Handoff contract

When handing artifacts to a consuming app, state explicitly:

- **What was regenerated** and from which checkpoint / commit SHA.
- **Measured parity** — correlation and max error vs the PyTorch reference.
- **Measured latency** and which compute units were used, on which chip.
- **WER/CER** on the eval set, and which eval set.
- **Input contract** — sample rate, feature dim, expected shapes, vocab size,
  and the tokenizer that must match
  ([utils/tokenizer.py](../../../utils/tokenizer.py)).
- **Known limitations** — sequence-length ceilings, bucket sizes, anything that
  falls back to CPU.

Read [docs/INTEGRATION_GUIDE.md](../../../docs/INTEGRATION_GUIDE.md) before
writing the handoff; it defines the app-side expectations.

## Procedure (agent)

1. Read [README.md](../../../README.md), [CLAUDE.md](../../../CLAUDE.md), and
   the relevant guide for the subsystem
   ([README/Mamba-on-Apple-Silicon.md](../../../README/Mamba-on-Apple-Silicon.md)
   for MPS/ANE,
   [docs/INTEGRATION_GUIDE.md](../../../docs/INTEGRATION_GUIDE.md) for handoff).
2. Establish which of the four "ship" meanings the user intends.
3. Run the [Pre-ship gate](#pre-ship-gate) before advising "ready to ship."
4. For **git remote** operations, follow **`git-push`**.
5. Report measured numbers, not adjectives.

## Anti-patterns

- Treating **green CI as a release gate** — it is a CPU smoke test.
- Shipping a `.mlpackage` without parity and placement evidence.
- Committing regenerated `.mlpackage` bundles — they are gitignored
  (`*.mlpackage/`) for a reason. Ship them out-of-band and record the
  provenance.
- Claiming a latency number from an unwarmed run, on battery, or without naming
  the chip.
- Reporting WER from the synthetic `--sanity` path. That path proves the code
  runs; it says nothing about accuracy.

## Related skills

- **`git-push`:** commit, merge, push, chase CI.
- **`git-commit`:** post-commit sanity-train awareness (non-blocking heads-up).
- **`coreml-validate`:** numerical parity before handoff.
- **`coreml-profile`:** compute-unit placement before handoff.
- **`bakeoff`:** controlled latency comparison across configurations.
- **`audit`:** full review when the user says **`audit`**.
