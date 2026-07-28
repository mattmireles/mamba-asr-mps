# v1 Ship: CTC-29 End-to-End On-Device ASR Plan

**Date:** 2026-07-27
**Status:** Stopped at Phase 2 accuracy gate — scale-up required (2026-07-27)

## Executive Summary

Produce the first working model this repo has ever had: a `vocab_size=29`
character-CTC ConMamba trained to convergence on LibriSpeech train-clean-100,
exported to Core ML with a numerical parity gate, and consumed by the Swift
runner with a single-sourced chunk/vocab contract. Before: no trained model
exists, real-audio WER has always been 1.000, and the export pipeline
green-lights random weights. After: measured test-clean WER from a Core ML
artifact running on this machine, with every claim backed by a command.

## Problem Statement

- **Symptom:** The pipeline shell executes end-to-end, but every transcript
  ever produced was garbage; the project has never worked.
- **Root Cause:** Three compounding failures found by the 2026-07-27 audit
  (see `CLAUDE.md` → Ground truth): the 1024-vs-29 vocab contract break
  (`README/implementation-plan-v2.md:337`), no model ever trained to
  convergence (NaN-at-~200-steps skipped, not solved), and no verification
  layer — no PyTorch↔Core ML parity check exists on any path.
- **Impact:** Every prior latency/optimization result was measured on models
  producing noise; all of that effort is unusable until a real model exists.

## Goals and Non-Goals

### Goals

- [ ] A `vocab_size=29` ConMambaCTC trained on train-clean-100 with greedy
      dev-clean WER ≤ 0.25 (stretch: ≤ 0.15), measured by
      `scripts/compute_wer_cer.py`.
- [x] A PyTorch↔Core ML parity harness that runs over a chunk *sequence* and
      fails loudly — proven on random weights **before** any training run.
- [x] One chunk/vocab/mel contract, written by the exporter and read by the
      Swift runner — a mismatch becomes a loud error, not a coincidence.
- [ ] Measured test-clean WER and per-chunk latency for the shipped
      `.mlpackage`, recorded in `README/training-notes.md` with machine and
      date.

### Non-Goals

- **RNN-T repair.** The loss backends are broken
  (`modules/rnnt_loss_mps.py:249`, `train_RNNT.py:706`); fixing them is a
  separate future plan, only if token-level streaming latency becomes a
  product requirement. Chunked CTC covers v1 streaming UX.
- **Phase-3 optimization (KD/QAT/pruning).** Cannot start before a converged
  model exists.
- **`ct.StateType` migration.** A refinement; plain tensor-I/O state is fine
  for v1 and is what the runner already speaks.
- **LM / beam-search decoding.** Greedy only. A decoding stack on top of a
  weak model is hand-engineering around a training problem.
- **iOS app integration.** v1 ships the artifact + Swift runner contract;
  `docs/INTEGRATION_GUIDE.md` gets updated, nothing more.

## Scope and Constraints

- **Scope:** CTC path only — `train.py`, `modules/Conmamba.py`, a CTC export
  wrapper, parity harness, Swift runner CTC mode, LibriSpeech eval.
- **Constraints:** Training runs locally on the M2 Ultra (64 GB). MPS
  `aten::_ctc_loss` falls back to CPU (`PYTORCH_ENABLE_MPS_FALLBACK=1`
  required). After duration bucketing and the parallel selective scan,
  measured full-corpus training compute was 25.7–59.7 minutes per epoch,
  excluding full-dev validation.
- **Guardrails:** Load-bearing docs stay put (`README/Mamba-on-Apple-Silicon.md`
  section numbers; `README/training-notes.md` and
  `README/implementation-plan-v2.md` are read at runtime by
  `scripts/report_phase3.sh` and `scripts/run_phase2_baselines.sh`).

## Ground Truth Contracts (Do Not Violate)

- **CI import constraint:** CI is `ubuntu-latest` + `requirements-ci.txt`
  (torch + numpy only). No new top-level import of `torchaudio`, `librosa`,
  `soundfile`, or `coremltools` may land on the `train_CTC.py --sanity` path.
  Real-audio imports stay lazy (inside functions) or module-guarded.
  `requirements-ci.txt` does not change in this plan.
- **Artifact policy (`README.md` → Artifact Policy):** no weights,
  `.mlpackage`, or datasets in git. The recovered 12-WAV eval set (~850 KB)
  is a *small test fixture* and is deliberately committed under
  `tests/fixtures/`.
- **Anti-rot duty:** every phase that changes a Ground-truth fact updates the
  `CLAUDE.md` Ground truth section in the same commit.
- **Honesty gate:** no phase reports a number it did not measure; "unverified"
  is the required word when a check was not run.

## Already Shipped (Do Not Re-Solve)

- **Selective scan kernel:** pure-PyTorch, MPS-clean logarithmic affine prefix
  scan. The Phase 2 real-data probe falsified the old "not a bottleneck"
  claim: the parallel graph plus duration bucketing and batch 8 increased a
  d256/6-block probe from 2.84 to 8.51 samples/s while retaining exact
  float64 forward/gradient parity with the literal recurrence.
- **CTC sanity gate:** green (exit 0, loss 55.4→42.2 on MPS, 3 s).
- **Swift runner:** builds clean; computes its own 80-mel log-spectrogram
  (n_fft 512, win 400, hop 160) in Swift, so DSP stays outside the graph.
- **Export mechanics:** `torch.jit.trace` → `mlprogram` → `coremlcompiler`
  → Swift streaming loop all execute on torch 2.8.0 + installed coremltools
  (proven 2026-07-27 with a random-weight MCT export, 3.17 ms/chunk CPU).
- **Eval tooling:** `scripts/compute_wer_cer.py` and the 12-WAV testset with
  references, recovered at
  `/Users/mm/Documents/Codex/2026-07-16/ho/work/tokei-github-audit-20260716-232821/trees/TranscendenceInc__MLX-fine-tuner/TranscendenceInc-MLX-fine-tuner-309c9ee874fb2d3d42955143e0df00c86d86b2cf/Mamba-ASR-MPS/exports/testset/`.

## Fresh Baseline (Current State)

- **Architecture:** ConMambaCTC (causal conv frontend ×2 stride-2 → 6×
  MambaBlock d256 → 29-logit linear head), 4× time reduction. Streaming
  carries eight mel frames plus each block's Mamba state.
- **Metrics:** a real d256/6-block train-clean-100 checkpoint now exists under
  gitignored `checkpoints/`, but it is not shippable. Independent full
  dev-clean evaluation measured CER `0.222494` and WER `0.612790` over 2,703
  utterances; the required WER is ≤ `0.25`.
- **Known gaps:** the audio environment and three required LibriSpeech splits
  are ready locally; direct CTC export, numerical parity, and the Swift
  contract are proven on random weights. Trained export, test-clean accuracy,
  latency, handoff, and release are blocked by the Phase 2 accuracy gate.

## Solution Overview

Order the work so everything cheap that could invalidate the expensive step
runs first. The full training run is the costly step; the contracts and
the parity harness are cheap and work on random weights — build and prove
them before training.

```
P0 env+data  -->  P1 contracts + parity (random weights)  -->  P2 train  -->  P3 export/eval/handoff
```

## Implementation Phases

> Do one phase at a time. Verify before proceeding. One commit per phase
> (`execute-plan` contract).

### Phase 0: Environment + Data (~1 day)

**Goal:** The real-audio path can run; LibriSpeech manifests exist; fixtures
are in-repo.

**Skills:** None of the domain skills apply — this is environment and data
plumbing. `phase-audit` still runs before commit (see `execute-plan`'s
per-phase loop).

**Tasks:**

- [x] Install audio stack into the active env, matching torch 2.8.0:
      `python3 -m pip install "torchaudio==2.8.*" librosa soundfile`.
- [x] (Optional, user-only) Check the locked legacy home
      `/Users/mattmireles/Documents/Training Data/LibriSpeech/` for an
      existing corpus copy before downloading. The path was not present on
      this machine, so the official archives were downloaded.
- [x] Download from OpenSLR (SLR12): `train-clean-100.tar.gz`,
      `dev-clean.tar.gz`, `test-clean.tar.gz`; extract under
      `data/LibriSpeech/` (gitignored).
- [x] Generate manifests: `python3 librispeech_prepare.py --data-dir
      data/LibriSpeech` → `path,duration,text` CSVs under `data/`.
- [x] Copy the recovered 12-WAV testset + refs into `tests/fixtures/testset/`
      (committed; ~850 KB) and note provenance in a short
      `tests/fixtures/README.md`.

**Verification:**

- `python3 -c "import torchaudio, librosa, soundfile"` → exit 0.
- Manifest row counts printed and ≈ expected (train-clean-100 ≈28.5k,
  dev-clean ≈2.7k, test-clean ≈2.6k utterances).
- Mechanical gate unchanged: `PYTHONPATH="$PWD"
  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 train_CTC.py --epochs 1 --sanity` →
  exit 0 (CI-equivalent still green; no new sanity-path imports).

**Recorded 2026-07-27:** imports exited 0
(`torchaudio 2.8.0`, `librosa 0.11.0`, `soundfile 0.13.1`);
train/dev/test manifests contain 28,539 / 2,703 / 2,620 rows with zero missing
paths, empty transcripts, or non-positive durations; the CI-equivalent sanity
run exited 0 with loss 36.2281 → 27.6546.

---

### Phase 1: Contracts + Verification Layer, Before Any Training (~2–3 days)

**Goal:** `vocab_size=29` end-to-end; a CTC export path with an explicit
state/chunk contract; a parity harness that passes on random weights. No
training yet.

**Skills:** `coreml-validate` — this phase's core deliverable
(`scripts/validate_parity.py`) *is* that skill's numerical-parity job, run
here on random weights instead of trained ones. Invoke it once the harness
exists to confirm the correlation/max-error methodology matches repo
convention before trusting the exit code.

**Tasks:**

- [x] Make 29 the default vocab everywhere: `modules/Conmamba.py`
      (`ConMambaConfig.vocab_size` 1024→29), `train.py` (train the 29-logit
      head directly; delete the 1024→29 projection-head training path),
      `train_CTC.py` (synthetic targets already fit in 29 — keep the sanity
      gate green).
- [x] New CTC streaming export in `scripts/export_coreml.py`: a
      `StreamingCTCWrapper` over ConMambaCTC — inputs
      `audio_chunk[1,C+8,80]` (eight carried mel frames + C new frames) +
      per-layer Mamba hidden states. Outputs are `logits[1,C/4,29]` +
      updated states. Plain
      tensor I/O (StateType is out of scope). Exporter writes
      `exports/contract.json`: chunk frames, time reduction, mel params,
      vocab list, state shapes.
- [x] Chunk-boundary policy, decided empirically, not speculatively: the
      parity harness (next task) measures chunked-vs-full-sequence
      divergence on real fixture audio. If greedy transcripts differ, carry
      conv left-context frames client-side; if not, accept and document the
      boundary approximation. Record the measurement in this plan's Debug
      Notes.
- [x] New `scripts/validate_parity.py`: runs the same fixture WAV through
      PyTorch (full sequence AND chunked-with-state) and Core ML (chunked),
      over ≥3 chunks; reports per-output correlation, max |Δ|, and greedy
      transcript equality; exits non-zero on tolerance failure (initial
      FP32 tolerance: corr ≥ 0.999, max |Δ| ≤ 1e-3 — tune only with recorded
      evidence).
- [x] Swift runner CTC mode in `swift/MambaASRRunner/Sources/.../main.swift`:
      read `contract.json` (chunk size, vocab, mel params, I/O names) instead
      of hardcoded constants; greedy CTC decode; keep the existing MCT flags
      working or explicitly retired.
- [x] Retire the projection tooling from the v1 path: mark
      `scripts/extract_projection_from_ckpt.py` and
      `scripts/make_projection_mod29.py` deprecated in their headers
      (deletion happens in Phase 3 cleanup).

**Verification:**

- `python3 scripts/export_coreml.py --output exports/ctc29_random.mlpackage`
  → exit 0 on random weights.
- `python3 scripts/validate_parity.py --mlpackage
  exports/ctc29_random.mlpackage --wav tests/fixtures/testset/audio/<one>.wav`
  → exit 0, correlation and max-error printed.
- Contract enforcement is loud: running the Swift runner with a mismatched
  `--chunk` fails with a clear message (demonstrate once, record output).
- `swift build -c release --package-path swift/MambaASRRunner` → exit 0.
- Mechanical gate: CTC sanity → exit 0.

**Recorded 2026-07-27:** default d256/6-block FP32 export exited 0
(`packageSHA256=100aed1ab73896d96e05164582e247340bfba6f2fd24b21d3ad6fafa7f3d9b30`).
On `utt_8.wav` tiled to three 256-frame chunks, aggregate PyTorch↔Core ML
logit correlation was 1.000000000 with max error `1.03712082e-05`; every
per-chunk logit/state correlation was 1.000000000, and full, chunked, and
Core ML greedy transcripts matched. Swift release build and contract-driven
real-WAV inference exited 0. `--chunk 512` against the 256-frame contract
failed before model compilation with exit 1 and
`CTC contract mismatch: --chunk 512 does not match contract chunkFrames 256`.
The CTC sanity gate exited 0 (loss 23.7023 → 12.8676).
The phase-local correctness, regression, integration, security, and scope
audit found no remaining Phase 1 blocker; `COORD-AGENTS.md` remained an
unrelated pre-existing worktree modification and was excluded.

---

### Phase 2: Train to Convergence (~2–3 days code + 12.6 h measured training compute)

**Goal:** A converged `checkpoints/best.pt` with dev-clean greedy WER ≤ 0.25.

**Skills:** `debug` — use it for the NaN-at-~200-steps root-cause task
specifically (systematic investigation, Context7 for any PyTorch/MPS
numerics question, a consolidated `write-notes` entry as its required final
step). Do not reach for it for the routine training-loop work, only the
stuck-and-unclear failure mode.

**Tasks:**

- [x] Root-cause the historical NaN-at-~200-steps on real data
      (`README/training-notes.md:15-20` recorded skipping, not solving).
      Time-box: reproduce within ~300 steps on train-clean-100, then
      investigate (suspects to check, not conclusions: CTC inf-cost samples
      where `T' < target_len`, LR warmup absence, MPS fallback numerics).
      Fix in `train.py` / `modules/`, document in Debug Notes.
- [x] Full run via `train.py`: train-clean-100 manifest, dev-clean
      validation each epoch, AdamW lr 3e-4, grad clip, checkpointing to
      `checkpoints/` (gitignored), metrics CSV committed to
      `README/training-notes.md` as a summarized table (not the raw CSV).
- [x] Stop rule: if two full runs plus one tuning round miss WER ≤ 0.25,
      STOP and write a scale-up decision memo (more data: +train-clean-360
      vs. deeper model) instead of silently grinding.

**Verification:**

- `python3 scripts/compute_wer_cer.py` over dev-clean greedy decodes of
  `checkpoints/best.pt` → printed WER ≤ 0.25 (record exact number, epoch,
  wall-clock, machine in `README/training-notes.md`).
- Loss curve summarized in `README/training-notes.md`; no NaN step-skips in
  the final run's log.
- Mechanical gate: CTC sanity → exit 0.

**Recorded 2026-07-27:** the independent evaluator and the executable corpus
scorer agreed on CER `0.222494` and WER `0.612790` over all 2,703 dev-clean
utterances (`52,677/236,757` character errors and `33,337/54,402` word
errors). The scorer's `--wer-threshold 0.25` gate exited 3. Two bounded
ten-epoch schedules completed with zero non-finite losses, zero non-finite
gradients, zero invalid samples, and zero empty valid batches. The stop rule
therefore fired; Phase 3 did not start. Current-code closeout gates passed:
MPS and forced-CPU CTC sanity, checkpoint/resume smoke, repeat-aware CTC and
sampler-continuity checks, float64 selective-scan forward/gradient parity,
MPS scan benchmark, and a fresh default d256/6 Core ML export
(`4107377e...`) with three-chunk correlation `1.000000000`, maximum logit
error `1.25169754e-05`, and matching full/chunked/Core ML transcripts.

---

### Phase 3: Export, Evaluate, Hand Off (~2–3 days)

**Goal:** A validated `.mlpackage` with measured accuracy and latency; docs
telling the truth; handoff per the `deploy` skill.

**Skills:** `coreml-validate` for the FP32/FP16 parity gate on trained
weights; `coreml-profile` to check actual compute-unit placement before
claiming ANE anywhere (replaces guessing from the Xcode "estimated" tab);
`bakeoff` for the `cpu`/`cpu-gpu`/`all` latency sweep — it already knows how
to prepare inputs, build the Swift runner, and record into
`README/training-notes.md`, so use it instead of hand-rolling the sweep;
`deploy` for the final handoff gate and release tagging.

**Tasks:**

- [ ] Export `best.pt` as FP32 and FP16 variants; run
      `scripts/validate_parity.py` on both (FP16 gate: greedy transcripts on
      the 12-WAV set match PyTorch within WER delta ≤ 0.005 absolute —
      task-metric judgment per `CLAUDE.md` Part 3.4, over a chunk sequence).
- [ ] Full-corpus eval: greedy WER/CER on test-clean through the **Swift
      runner** (refresh `scripts/eval_batch.sh` for the CTC contract), plus
      the 12-WAV fixture set. Record in `README/training-notes.md`.
- [ ] Latency sweep on this machine: ms/chunk and RTF for `cpu`, `cpu-gpu`,
      `all` at the contract chunk size (runner `--latency-csv`); claim ANE
      placement only if demonstrated (route through the `coreml-profile`
      skill; otherwise write "not demonstrated").
- [ ] Hygiene: delete `scripts/extract_projection_from_ckpt.py`,
      `scripts/make_projection_mod29.py`, and the projection-head remnants;
      fix stale doc claims (`docs/INTEGRATION_GUIDE.md` `test_integration.py`
      reference; README "targets the ANE" → measured statement); update
      `CLAUDE.md` Ground truth (trained model now exists; export settings;
      parity gate exists).
- [ ] Handoff per `deploy` skill: tag the release, document artifact
      regeneration commands, update `docs/INTEGRATION_GUIDE.md` with the
      contract.json flow.

**Verification:**

- `python3 scripts/validate_parity.py --mlpackage exports/MambaASR_ctc29_fp16.mlpackage ...`
  → exit 0.
- Test-clean WER printed by `scripts/compute_wer_cer.py` from Swift-runner
  transcripts; number recorded with date + machine.
- Latency table in `README/training-notes.md` with all three compute modes.
- `git grep -l 'test_integration.py' docs/` → no hits.
- Mechanical gate + `swift build` → exit 0.

## Executable Memory

- Regression test (contract + numerics):
  `python3 scripts/validate_parity.py --mlpackage <pkg> --wav tests/fixtures/testset/audio/<one>.wav`
- Regression test (accuracy): `python3 scripts/compute_wer_cer.py
  --predictions <pred.txt> --references <ref.txt>`
- Not testable by command: the chunk-boundary policy decision — proven by the
  recorded divergence measurement in Debug Notes.

## Success Criteria

### Hard Requirements (Must Pass)

- [ ] Parity harness exits 0 on the shipped FP16 artifact over a ≥3-chunk
      sequence of real audio.
- [ ] Test-clean greedy WER ≤ 0.25 measured end-to-end through the Swift
      runner (stretch ≤ 0.15).
- [ ] RTF < 0.3 in the best compute mode at the contract chunk size
      (baseline evidence: 3.17 ms per 2.56 s chunk on random weights —
      enormous headroom; the gate exists to catch regressions).
- [ ] CI green on every phase commit; `requirements-ci.txt` untouched.
- [ ] No weights, datasets, or `.mlpackage` in git.

### Definition of Done

- [ ] All four phase verifications recorded (exit codes + numbers) in
      `README/training-notes.md`.
- [x] `CLAUDE.md` Ground truth section reflects the new reality through the
      Phase 2 stop.
- [ ] Release tagged; `docs/INTEGRATION_GUIDE.md` matches the shipped
      contract.
- [ ] Follow-up plans (RNN-T repair, Phase-3 optimization, StateType) listed
      as candidates, deliberately unwritten.

## Open Questions

### Resolved

- **Q:** CTC-first or fix RNN-T first? **A:** CTC-first — it trains today;
  RNN-T is the largest broken surface and v1 does not need it (2026-07-27
  assessment, ratified by plan invocation).
- **Q:** Which corpus? **A:** train-clean-100 first. The Phase 2 stop rule
  fired at WER `0.612790`; the next experiment should add train-clean-360
  while holding d256/6 fixed.
- **Q:** Decoding? **A:** Greedy only for v1.
- **Q:** More data or a deeper model after the Phase 2 miss? **A:** More data
  first. Train loss continued down while dev loss and WER plateaued, so
  train-clean-360 is the simpler test of the observed generalization gap.
  Reconsider capacity only after that controlled data-scale run.

### Unresolved

- **Q:** FP16 or FP32 as the shipped default? Decided by Phase 3 parity +
  latency measurements, deferred until a future checkpoint passes the
  accuracy gate.

## References

### Internal

- `CLAUDE.md` → Ground truth (audited 2026-07-27) — the factual baseline for
  this plan.
- [training-notes.md](../training-notes.md) — NaN history, throughput, prior
  latency tables.
- [implementation-plan-v2.md](../implementation-plan-v2.md) — predecessor
  roadmap; §Accuracy sanity documents the vocab-mismatch root cause.
- [Mamba-on-Apple-Silicon.md](../Mamba-on-Apple-Silicon.md) — MPS/ANE
  constraints (cited by section number from code; do not renumber).
- [plan-workflow-skills-guide.md](../Skills/plan-workflow-skills-guide.md) —
  the mechanical gate and execute-plan loop.

### External

- OpenSLR SLR12 — LibriSpeech corpus: <https://www.openslr.org/12>

## Modules

### Phase Dependencies

```
P0 --> P1 --> P2 --> P3
```

Strictly serial. P1 is deliberately before P2: the parity harness works on
random weights, so export correctness is proven before a week of training is
spent on it.

### Files Likely to Change

| File | Change Type | Notes |
| --- | --- | --- |
| `modules/Conmamba.py` | Modify | `vocab_size` default 1024→29 |
| `train.py` | Modify | Train 29-logit head directly; drop projection path; NaN fix |
| `train_CTC.py` | Modify | Keep sanity gate green under vocab 29 |
| `scripts/export_coreml.py` | Modify | CTC `StreamingCTCWrapper` + `contract.json` |
| `scripts/validate_parity.py` | Create | PyTorch↔Core ML chunk-sequence parity gate |
| `scripts/eval_batch.sh` | Modify | CTC contract; test-clean batch eval |
| `swift/MambaASRRunner/.../main.swift` | Modify | Read `contract.json`; CTC greedy mode |
| `tests/fixtures/testset/` | Create | 12-WAV eval fixtures + refs (committed) |
| `scripts/extract_projection_from_ckpt.py` | Delete | Phase 3; projection path retired |
| `scripts/make_projection_mod29.py` | Delete | Phase 3; projection path retired |

### Performance and Latency Budget

| Operation | Target | Current |
| --- | --- | --- |
| Per-chunk inference (contract chunk) | < 50 ms | 3.17 ms (random weights, CPU, 2026-07-27) |
| RTF, best compute mode | < 0.3 | unmeasured on trained weights |
| Training epoch, train-clean-100 | bounded local run | 25.7–59.7 min training compute, excluding validation |

### Risks and Mitigations

- **Historical NaN premise:** resolved as an RNN-T incident, not direct CTC.
  The CTC loop now enforces repeat-aware feasibility and rejects non-finite
  losses or gradients explicitly.
- **CTC-loss CPU fallback throughput:** mitigated by duration bucketing,
  batch 8, and the parallel selective scan; both ten-epoch runs completed
  locally without a numerical failure.
- **Chunk-boundary degradation in streaming CTC:** conv frontend overlap may
  hurt at boundaries → Phase 1 measures it; carry left-context only if the
  measurement demands it.
- **WER landed above 0.25:** the stop rule fired. Add train-clean-360 while
  holding architecture and evaluation fixed; do not deepen the model or start
  Phase 3 in this plan.
- **Doc-runtime coupling:** `scripts/report_phase3.sh` reads
  `README/training-notes.md` at runtime → append, don't restructure, when
  recording results.

### Progress Tracker

#### Phase 0: Environment + Data

- [x] Audio stack installed and importable
- [x] LibriSpeech downloaded, manifests generated
- [x] Fixtures committed under `tests/fixtures/testset/`

#### Phase 1: Contracts + Verification Layer

- [x] vocab 29 default end-to-end, sanity gate green
- [x] CTC export + `contract.json`
- [x] `validate_parity.py` passing on random weights
- [x] Swift runner CTC mode reading the contract

#### Phase 2: Train to Convergence

- [x] NaN root-caused and current CTC loop hardened (Debug Notes entry)
- [x] Two ten-epoch schedules complete; exact full-dev metrics recorded
- [x] Stop rule applied; train-clean-360 chosen before deeper capacity
- [ ] dev-clean WER ≤ 0.25 (failed: best `0.612790`)

#### Phase 3: Export, Evaluate, Hand Off

- [ ] FP16 parity gate passed
- [ ] Test-clean WER + latency table recorded
- [ ] Projection tooling deleted; docs de-rotted; Ground truth updated
- [ ] Release tagged and handed off

### Debug Notes

Append real issues encountered during implementation with fixes.

- **2026-07-27 — Phase 0:** the plan's one-command
  `librispeech_prepare.py --data-dir ...` interface did not exist, and `data/`
  was not ignored. Added the compatible three-split command, retained the
  legacy `--root/--split` path, and added `data/` to `.gitignore`. The primary
  OpenSLR mirror stalled at 79% of train-clean-100; a resumable transfer via
  the official `openslr.trmal.net` mirror completed the same archive. Published
  MD5s matched before extraction:
  `2a93770f6d5c6c964bc36631d331a522` (train),
  `42e2234ba48799c1f50f24a7926300a1` (dev), and
  `32fa31d27d2e1cad72775fee3f4849a9` (test).
- **2026-07-27 — Phase 1 boundary policy:** the first default-size random
  gate falsified chunk-local symmetric convolution: PyTorch↔Core ML numerics
  passed, but full and chunked greedy transcripts differed. Replaced future-
  dependent symmetric padding with a causal, bias-free two-convolution
  frontend. The client now carries eight mel frames (six-frame receptive
  history rounded to the four-frame stride), and the model discards the two
  context-only outputs. Full/chunked PyTorch logits then agreed to
  `6.7e-07`; the default-size three-chunk transcript gate passed.
- **2026-07-27 — Phase 1 Swift DSP/state:** Swift initially produced NaNs
  because new `MLMultiArray` state storage was not guaranteed initialized;
  the CTC path now explicitly zero-fills it. Swift also placed the 400-sample
  Hann window at the start of each 512-point frame, while `torch.stft`
  centers a shorter window inside the FFT frame even with `center=false`.
  Centering the window yielded Swift↔Python mel correlation 1.000000000,
  max error `5.13e-04`, on the first 64 frames of `utt_8.wav`.
- **2026-07-27 — Phase 2 historical NaN:** the cited evidence at
  `README/training-notes.md:15-20` describes an RNN-T `T′·U` alignment-grid
  failure, not the direct CTC trainer. A seeded d256/6-block CTC run reached
  300 real MPS batches with loss 3.3254 → 2.4990 and zero non-finite losses;
  a CPU probe matched the early loss. All 28,539 train-clean-100 utterances
  satisfy the repeat-aware CTC length bound. `train.py` now makes that bound
  explicit, exposes rather than zeros infinite loss, rejects non-finite
  gradient norms before the optimizer step, crops padded validation outputs,
  reports WER, and writes resumable checkpoints/metrics. Context7 confirmed
  PyTorch 2.8's `zero_infinity` behavior.
- **2026-07-27 — Phase 2 training throughput:** the full real-data probe
  falsified the inherited claim that selective scan was not a training
  bottleneck. Replacing the Python timestep recurrence with an exact
  logarithmic affine prefix scan cut a 200-sample d256/6-block MPS probe from
  70.5 seconds to 26.8 seconds; duration bucketing plus batch 8 reduced it to
  23.5 seconds. Float64 forward/gradient comparison against the literal
  recurrence agreed to `8.88e-16`, and a converted Core ML smoke model passed
  three-chunk parity (correlation 1.0, max logit error `1.67e-06`, matching
  transcript). Batch 16 regressed, so the full run uses batch 8.
- **2026-07-27 — Phase 2 run 1 miss:** ten epochs completed with zero
  non-finite losses, zero non-finite gradients, and zero invalid samples.
  The independent full-manifest scorer measured dev-clean CER `0.222494` and
  WER `0.612790` over all 2,703 utterances, so the `0.25` WER gate failed.
  The dataset's legacy 20-second default had silently capped the trainer's
  first validation view at 2,642 utterances; v1 now explicitly keeps every
  manifest row. The cosine LR reached zero while training loss was still
  falling, so run 2 isolates that hypothesis by resuming the best weights
  with a fresh optimizer and constant `1e-4`, without simultaneously changing
  model size or corpus.
- **2026-07-27 — Phase 2 stop:** run 2 completed epochs 11–20 with a constant
  `1e-4`, zero non-finite losses, zero non-finite gradients, zero invalid
  samples, and zero empty valid batches. Train loss fell `0.5698 → 0.4515`,
  but dev loss ended at `0.9262` and WER never beat run 1's independently
  measured `0.612790`. The standalone accuracy gate exited 3 at threshold
  `0.25`. Per the stop rule, Phase 3 is blocked. The next controlled
  experiment is train-clean-100 + train-clean-360 with d256/6 held fixed;
  deeper capacity is deferred until data scaling is measured.
- **2026-07-27 — Phase 2 resume order:** the machine restart after epoch 16
  restored model, optimizer, scheduler, and global step, but the
  duration-bucket sampler's in-memory epoch counter restarted at zero. Epochs
  17–20 therefore reused the shuffle-seed sequence from epochs 11–14 while
  still covering every training utterance once per epoch. Future resumes now
  initialize the sampler counter from the checkpoint epoch.

---

## Critical Reminder

> SIMPLER IS BETTER. If you are adding complexity, justify it. Most of the
> time, the simplest solution wins.
