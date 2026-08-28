---
name: create-plan
description: Create an implementation plan for this repo. Use when the user wants a checked-in plan under README/Plans, wants the work scoped into phases, and expects the plan to be built from repo guides and notes first, with Context7 only when current external library or framework behavior materially affects the plan. Do not use for implementing the work, informal brainstorming, or lightweight notes.
---

# Create Plan

## Template Contract (Non-Negotiable)

`assets/Plans-template.md` is this skill's canonical plan skeleton. It is a
build input, not optional reference material.

Every numbered phase (`### Phase N:`) must include a **Skills:** line naming
existing skills to read before that phase starts.

After choosing the target path, create the draft only by running:

```sh
scripts/scaffold-plan.sh <new-plan-path>
```

Fill that generated file; do not hand-write a competing skeleton. Preserve all
required headings in their existing order. Content under `## Modules` is
optional and may be selected or omitted only when it is not relevant. Before
handoff, run:

```sh
scripts/validate-plan.sh <new-plan-path>
```

The validator rejects a missing, renamed, or reordered required heading,
unfilled top-level placeholders, and any phase without **Skills:**. Use
`--allow-placeholders` only to check a fresh scaffold before drafting.

## Purpose

Use this skill to turn a concrete request into a repo-native implementation
plan. The output is a real plan file under `README/Plans/...`, not a chat-only
outline.

## Use When

- The work is large enough to need a real implementation plan.
- The user wants a plan written into the repo.
- The implementation needs phases, verification steps, and concrete files.

## Do Not Use When

- The user wants direct implementation instead of planning.
- The request is still too vague to scope honestly.
- The output should be a note, scratchpad, or brainstorm instead of a plan.

## Procedure

1. Read [references/index.md](references/index.md) first.
2. Gather repo context in this order:
   - the two existing roadmaps —
     [implementation-plan-v2.md](../../../README/implementation-plan-v2.md) is
     current, [implementation-plan.md](../../../README/implementation-plan.md)
     is the original Phase 1–4 plan
   - directly related guides in
     [README/Guides](../../../README/Guides) and [docs/](../../../docs)
   - directly related notes —
     [training-notes.md](../../../README/training-notes.md),
     [coreml-telemetry-issues.md](../../../README/coreml-telemetry-issues.md),
     and anything in [README/Notes](../../../README/Notes)
   - [README/Skills](../../../README/Skills) when the work touches plan
     workflow, skills, or routing
   - neighboring plans in the target `README/Plans/...` subtree
3. Use Context7 only when the plan depends on current library, framework, or API
   behavior that may have changed — `coremltools`, PyTorch MPS, `torchaudio`,
   Core ML deployment targets.
4. If Context7 is insufficient, fall back to official vendor docs.
5. Choose the most specific existing `README/Plans/...` subdirectory that fits
   the work. If none fits cleanly, place the plan in the closest higher-level
   subtree instead of inventing a noisy new folder.
6. Fill the file created by `scripts/scaffold-plan.sh` and make it
   implementation ready:
   - concrete phases
   - **Skills:** on every phase, naming existing skills the implementer must
     read before starting that phase
   - specific files where knowable
   - verification per phase — a **real command with an observable result**
   - hard requirements
   - rollback or kill switch when relevant
7. Audit the draft before stopping:
   - no missing policy that an implementer would have to invent
   - no fake certainty where the repo context is incomplete
   - no implementation work performed while planning

## Verification must be real

This repo has **no test suite**. A phase whose verification says "run the tests"
is not implementation ready. Use the actual gate:

```bash
PYTHONPATH="$PWD" PYTORCH_ENABLE_MPS_FALLBACK=1 python3 train_CTC.py --epochs 1 --sanity
```

Per-surface checks are tabulated in
[The Mechanical Gate](../../../README/Skills/plan-workflow-skills-guide.md#the-mechanical-gate-repo-specific).
If a phase should add tests, say so as phase scope — then `python3 -m pytest`
becomes a real gate for later phases.

## CI constraint to design around

CI runs `train_CTC.py --epochs 1 --sanity` on **`ubuntu-latest`, CPU only**,
with only [requirements-ci.txt](../../../requirements-ci.txt) installed (torch +
numpy). Any plan that puts a new import of `coremltools`, `librosa`,
`soundfile`, or `torchaudio` on the CTC sanity path must either keep it lazy or
budget a phase for updating `requirements-ci.txt`. Call this out in the plan.

## Canonical Docs

Read [references/index.md](references/index.md) first. It maps the canonical
workflow guide, the plan template, and the repo references to inspect before
writing a new plan.

## Handoff Rules

- Hand off to `execute-plan` only after the plan is checked in and the user
  wants implementation.
- Hand off to a domain skill only if the planning request narrows into a
  domain-specific technical question that must be answered before the plan can
  be completed.
