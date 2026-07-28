---
name: bakeoff
description: Run controlled mamba-asr-mps accuracy or runtime comparisons against frozen inputs and baselines. The recorded 2026-07-27 WER is 0.612790 against a 0.25 gate, so the default outcome is KILL and no Phase 3 or release progression unless a new preregistered experiment explicitly supersedes that result.
---

# Mamba ASR bakeoff

Freeze corpus, revisions, preprocessing, decoding, seeds, metrics, thresholds,
and artifact paths before execution. Compare the simplest reference with one
candidate change at a time.

## Hard stop

The current evidence fails the accuracy gate:

- measured WER: `0.612790`
- required WER: `<= 0.25`
- decision: `KILL`

Do not proceed to Phase 3, performance promotion, or release on the basis of
speed while this gate fails. Preserve the negative result and require a new
explicit hypothesis and frozen gate before rerunning.
