---
name: coreml-validate
description: >-
  Validate mamba-asr-mps Core ML exports against the PyTorch or Hugging Face
  reference. Use for numerical parity, explicit recurrent tensor I/O,
  encoder/decoder state propagation, transcript comparison, or export
  regressions. Do not require ct.StateType: this repo uses explicit tensors.
---

# Core ML validation

1. Read `CLAUDE.md`, the current export path, and validation fixtures.
2. Treat recurrent state as explicit model inputs and outputs.
3. Run identical frozen inputs through the reference and Core ML paths.
4. Compare tensor shape, dtype, max/mean error, correlation, and transcript
   behavior at the same state boundary.
5. Fail closed on missing or reordered explicit state tensors.
6. Record model/package hashes with results.

Never “fix” validation by inventing `ct.StateType` or weakening thresholds after
seeing the result.
