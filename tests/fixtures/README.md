# Real-Audio ASR Fixtures

`testset/` contains the small, committed real-audio corpus used by the CTC
export, parity, and Swift-runner gates. Each WAV has a same-stem reference
transcript under `testset/refs/`.

## Provenance

The 12 WAV/reference pairs were recovered on 2026-07-27 from the audited
repository snapshot at:

```text
/Users/mm/Documents/Codex/2026-07-16/ho/work/
tokei-github-audit-20260716-232821/trees/
TranscendenceInc__MLX-fine-tuner/
TranscendenceInc-MLX-fine-tuner-309c9ee874fb2d3d42955143e0df00c86d86b2cf/
Mamba-ASR-MPS/exports/testset/
```

The snapshot name records source commit
`309c9ee874fb2d3d42955143e0df00c86d86b2cf`. The fixtures are 16 kHz mono WAV
files and are intentionally versioned; full datasets, checkpoints, and Core ML
packages remain excluded by the repository artifact policy.

## Known Duplication

`hello_world_.wav`, `hello_world_16k.wav`, and `utt_1.wav` are byte-identical
recovered copies with the same `hello world` reference. They remain separate so
the fixture set is an exact, traceable copy of the recovered 12-file gate.
