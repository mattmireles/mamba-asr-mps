# mamba-asr-mps reading list (persona + training/export work)

Paths are **repo-root relative** (open from the `mamba-asr-mps` checkout root).

## Canonical persona and playbook

- `CLAUDE.md` — Ilya persona, MPS training constraints, PyTorch → Core ML
  checklist, ANE layout, divide-and-conquer

## Architecture and models

- `modules/Conmamba.py` — ConMambaCTC: conv frontend + Mamba encoder + CTC head
- `modules/mamba/` — selective scan / SSM core
- `modules/mct/` — Mamba-Conformer-Transducer, streaming RNN-T
- `modules/rnnt_loss.py`, `modules/rnnt_loss_mps.py` — the two RNN-T loss paths
- `docs/SYSTEM_ARCHITECTURE.md` — module boundaries and data flow
- `docs/mamba-asr-landscape.md` — research context and architecture rationale

## Training

- `train_CTC.py` — CTC entry point; `--sanity` is the repo's mechanical gate
- `train_RNNT.py` — RNN-T entry point
- `train.py` — shared/legacy training path with the 1024→29 projection head
- `hparams/{CTC,RNNT,S2S}/` — configurations
- `config/apple_silicon_config.py`, `config/environment_config.py`
- `README/training-notes.md` — run history, loss curves, what diverged

## Apple silicon runtime and performance

- `README/Mamba-on-Apple-Silicon.md` — MPS/ANE strategy and targets (cited by
  **section number** from training and benchmark code — do not renumber)
- `docs/Mamba-Apple-Silicon-guide.md` — PyTorch MPS semantics, deployment
- `benchmarks/bench_mps.py`, `benchmarks/bench_selective_scan.py`
- `scripts/bench_rnnt_impls.py`, `scripts/bench_selective_scan_report.py`
- `utils/hardware.py` — device detection

## Export and deployment

- `scripts/export_coreml.py` — stateful Core ML export, ANE targeting
- `scripts/export_and_validate.py` — export → compile → Swift validation
- `scripts/coreml_ops_remediation.py` — op-level fixes already applied
- `scripts/optimize.py` — distillation, QAT, pruning (Phase 3)
- `README/coreml-telemetry-issues.md` — export and ANE placement institutional memory
- `swift/MambaASRRunner/` — Swift host, CTC beam search, latency CSV
- `docs/INTEGRATION_GUIDE.md` — iOS/macOS integration contract

## Measurement

- `scripts/run_latency_probe.sh`, `scripts/summarize_latency_csv.py`
- `scripts/summarize_chunk_sweep.py`, `scripts/compare_sweeps.py`
- `scripts/compute_wer_cer.py` — the metric that actually decides shipping
- `scripts/compare_models_cpu.py` — reads the CPU baseline table out of
  `README/implementation-plan-v2.md`

## Roadmaps

- `README/implementation-plan-v2.md` — current; Phase 3 optimization
- `README/implementation-plan.md` — original Phase 1–4

## Repo process (plan-driven work)

- `README/Skills/plan-workflow-skills-guide.md`
- `README/Skills/phase-audit-rubric.md`

## Related skills (narrower charters)

- `.claude/skills/audit/SKILL.md` — findings-first review; word **audit**
- `.claude/skills/debug/SKILL.md` — systematic defect investigation
- `.claude/skills/execute-plan/SKILL.md` — phased plan execution
- `.claude/skills/coreml-validate/SKILL.md` — numerical parity
- `.claude/skills/coreml-profile/SKILL.md` — compute-unit placement
- `.claude/skills/bakeoff/SKILL.md` — controlled latency + WER comparison
