# v1 Ship: CTC-29 End-to-End On-Device ASR Plan

**Date:** 2026-07-27
**Status:** Planned

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
- [ ] A PyTorch↔Core ML parity harness that runs over a chunk *sequence* and
      fails loudly — proven on random weights **before** any training run.
- [ ] One chunk/vocab/mel contract, written by the exporter and read by the
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
  required); measured synthetic throughput is 42.9% of the repo's target, so
  budget ~2–4 h/epoch on 100 h of audio (estimate, verify in Phase 2).
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

- **Selective scan kernel:** pure-PyTorch, MPS-clean, 22–24k tok/s flat from
  seq 256→8192 (`benchmarks/bench_selective_scan.py`, 2026-07-27). Not a
  bottleneck; do not hand-optimize it in this plan.
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

- **Architecture:** ConMambaCTC (conv frontend ×2 stride-2 → 6× MambaBlock
  d256 → linear head), 4× time reduction. Sound; unchanged in this plan.
- **Metrics:** no trained checkpoint exists anywhere on this machine; best
  historical accuracy CER ≈0.86 on synthetic audio; real WER always 1.000.
- **Known gaps:** active `python3` lacks `torchaudio`/`librosa`/`soundfile`;
  no LibriSpeech copy on disk; exporter exports the wrong model (MCT) for the
  CTC path; chunk=256 agreement between exporter and Swift is coincidental.

## Solution Overview

Order the work so everything cheap that could invalidate the expensive step
runs first. The week-long training run is the costly step; the contracts and
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

**Tasks:**

- [ ] Install audio stack into the active env, matching torch 2.8.0:
      `python3 -m pip install "torchaudio==2.8.*" librosa soundfile`.
- [ ] (Optional, user-only) Check the locked legacy home
      `/Users/mattmireles/Documents/Training Data/LibriSpeech/` for an
      existing corpus copy before downloading.
- [ ] Download from OpenSLR (SLR12): `train-clean-100.tar.gz`,
      `dev-clean.tar.gz`, `test-clean.tar.gz`; extract under
      `data/LibriSpeech/` (gitignored).
- [ ] Generate manifests: `python3 librispeech_prepare.py --data-dir
      data/LibriSpeech` → `path,duration,text` CSVs under `data/`.
- [ ] Copy the recovered 12-WAV testset + refs into `tests/fixtures/testset/`
      (committed; ~850 KB) and note provenance in a short
      `tests/fixtures/README.md`.

**Verification:**

- `python3 -c "import torchaudio, librosa, soundfile"` → exit 0.
- Manifest row counts printed and ≈ expected (train-clean-100 ≈28.5k,
  dev-clean ≈2.7k, test-clean ≈2.6k utterances).
- Mechanical gate unchanged: `PYTHONPATH="$PWD"
  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 train_CTC.py --epochs 1 --sanity` →
  exit 0 (CI-equivalent still green; no new sanity-path imports).

---

### Phase 1: Contracts + Verification Layer, Before Any Training (~2–3 days)

**Goal:** `vocab_size=29` end-to-end; a CTC export path with an explicit
state/chunk contract; a parity harness that passes on random weights. No
training yet.

**Tasks:**

- [ ] Make 29 the default vocab everywhere: `modules/Conmamba.py`
      (`ConMambaConfig.vocab_size` 1024→29), `train.py` (train the 29-logit
      head directly; delete the 1024→29 projection-head training path),
      `train_CTC.py` (synthetic targets already fit in 29 — keep the sanity
      gate green).
- [ ] New CTC streaming export in `scripts/export_coreml.py`: a
      `StreamingCTCWrapper` over ConMambaCTC — inputs `audio_chunk[1,C,80]` +
      per-layer Mamba hidden states; outputs `logits[1,C/4,29]` + updated
      states. Plain tensor I/O (StateType is out of scope). Exporter writes
      `exports/contract.json`: chunk frames, time reduction, mel params,
      vocab list, state shapes.
- [ ] Chunk-boundary policy, decided empirically, not speculatively: the
      parity harness (next task) measures chunked-vs-full-sequence
      divergence on real fixture audio. If greedy transcripts differ, carry
      conv left-context frames client-side; if not, accept and document the
      boundary approximation. Record the measurement in this plan's Debug
      Notes.
- [ ] New `scripts/validate_parity.py`: runs the same fixture WAV through
      PyTorch (full sequence AND chunked-with-state) and Core ML (chunked),
      over ≥3 chunks; reports per-output correlation, max |Δ|, and greedy
      transcript equality; exits non-zero on tolerance failure (initial
      FP32 tolerance: corr ≥ 0.999, max |Δ| ≤ 1e-3 — tune only with recorded
      evidence).
- [ ] Swift runner CTC mode in `swift/MambaASRRunner/Sources/.../main.swift`:
      read `contract.json` (chunk size, vocab, mel params, I/O names) instead
      of hardcoded constants; greedy CTC decode; keep the existing MCT flags
      working or explicitly retired.
- [ ] Retire the projection tooling from the v1 path: mark
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

---

### Phase 2: Train to Convergence (~2–3 days code + ~1 week wall-clock)

**Goal:** A converged `checkpoints/best.pt` with dev-clean greedy WER ≤ 0.25.

**Tasks:**

- [ ] Root-cause the historical NaN-at-~200-steps on real data
      (`README/training-notes.md:15-20` recorded skipping, not solving).
      Time-box: reproduce within ~300 steps on train-clean-100, then
      investigate (suspects to check, not conclusions: CTC inf-cost samples
      where `T' < target_len`, LR warmup absence, MPS fallback numerics).
      Fix in `train.py` / `modules/`, document in Debug Notes.
- [ ] Full run via `train.py`: train-clean-100 manifest, dev-clean
      validation each epoch, AdamW lr 3e-4, grad clip, checkpointing to
      `checkpoints/` (gitignored), metrics CSV committed to
      `README/training-notes.md` as a summarized table (not the raw CSV).
- [ ] Stop rule: if two full runs plus one tuning round miss WER ≤ 0.25,
      STOP and write a scale-up decision memo (more data: +train-clean-360
      vs. deeper model) instead of silently grinding.

**Verification:**

- `python3 scripts/compute_wer_cer.py` over dev-clean greedy decodes of
  `checkpoints/best.pt` → printed WER ≤ 0.25 (record exact number, epoch,
  wall-clock, machine in `README/training-notes.md`).
- Loss curve summarized in `README/training-notes.md`; no NaN step-skips in
  the final run's log.
- Mechanical gate: CTC sanity → exit 0.

---

### Phase 3: Export, Evaluate, Hand Off (~2–3 days)

**Goal:** A validated `.mlpackage` with measured accuracy and latency; docs
telling the truth; handoff per the `deploy` skill.

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
- [ ] `CLAUDE.md` Ground truth section reflects the new reality.
- [ ] Release tagged; `docs/INTEGRATION_GUIDE.md` matches the shipped
      contract.
- [ ] Follow-up plans (RNN-T repair, Phase-3 optimization, StateType) listed
      as candidates, deliberately unwritten.

## Open Questions

### Resolved

- **Q:** CTC-first or fix RNN-T first? **A:** CTC-first — it trains today;
  RNN-T is the largest broken surface and v1 does not need it (2026-07-27
  assessment, ratified by plan invocation).
- **Q:** Which corpus? **A:** train-clean-100 first; scale is a Phase 2 stop-
  rule decision, not a default.
- **Q:** Decoding? **A:** Greedy only for v1.

### Unresolved

- **Q:** Is WER ≈ 0.20–0.25 acceptable for the intended product use?
  **Options:** ship as v1 demo / hold for train-clean-360 / hold for LM
  decoding. Lean: ship the demo, decide with the real number in hand.
- **Q:** FP16 or FP32 as the shipped default? Decided by Phase 3 parity +
  latency measurements.

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
| Training epoch, train-clean-100 | ~2–4 h (estimate) | unmeasured |

### Risks and Mitigations

- **NaN root cause is unknown:** could stall Phase 2 → time-boxed
  investigation with the `debug` skill; suspects listed, not presumed;
  escalate to a scale/precision decision memo rather than grinding.
- **CTC-loss CPU fallback throughput (42.9% of target):** epochs may run
  long → measure epoch 1 wall-clock, tune batch size via
  `config/apple_silicon_config.py`, accept longer wall-clock before adding
  complexity.
- **Chunk-boundary degradation in streaming CTC:** conv frontend overlap may
  hurt at boundaries → Phase 1 measures it; carry left-context only if the
  measurement demands it.
- **WER lands above 0.25:** stop rule triggers a deliberate scale-up decision
  (train-clean-360 vs. deeper model) with the M2 Ultra able to absorb either.
- **Doc-runtime coupling:** `scripts/report_phase3.sh` reads
  `README/training-notes.md` at runtime → append, don't restructure, when
  recording results.

### Progress Tracker

#### Phase 0: Environment + Data

- [ ] Audio stack installed and importable
- [ ] LibriSpeech downloaded, manifests generated
- [ ] Fixtures committed under `tests/fixtures/testset/`

#### Phase 1: Contracts + Verification Layer

- [ ] vocab 29 default end-to-end, sanity gate green
- [ ] CTC export + `contract.json`
- [ ] `validate_parity.py` passing on random weights
- [ ] Swift runner CTC mode reading the contract

#### Phase 2: Train to Convergence

- [ ] NaN root-caused and fixed (Debug Notes entry)
- [ ] Full run complete; dev-clean WER ≤ 0.25 recorded

#### Phase 3: Export, Evaluate, Hand Off

- [ ] FP16 parity gate passed
- [ ] Test-clean WER + latency table recorded
- [ ] Projection tooling deleted; docs de-rotted; Ground truth updated
- [ ] Release tagged and handed off

### Debug Notes

Append real issues encountered during implementation with fixes.

---

## Critical Reminder

> SIMPLER IS BETTER. If you are adding complexity, justify it. Most of the
> time, the simplest solution wins.
