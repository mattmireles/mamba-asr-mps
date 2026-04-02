#!/usr/bin/env bash
set -euo pipefail
# One-touch Phase 3 report: latency probe sweep + RNNT baselines + doc updates

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COREML_DIR="$ROOT_DIR/exports/CoreMLTraces"
EXPORTS_DIR="$ROOT_DIR/exports"
PLAN_MD="$ROOT_DIR/README/implementation-plan-v2.md"
NOTES_MD="$ROOT_DIR/README/training-notes.md"

# 1) Latency sweep (compute modes and chunk)
CHUNKS_ENV=${LAT_SWEEP_CHUNKS:-256}
MODES_ENV=${LAT_SWEEP_MODES:-all,cpu,cpu-gpu}
LAT_DURATION=${LAT_DURATION:-10}
LAT_WARMUP=${LAT_WARMUP:-2}
export LAT_SWEEP_CHUNKS="$CHUNKS_ENV"
export LAT_SWEEP_MODES="$MODES_ENV"
export LAT_DURATION
export LAT_WARMUP

bash "$ROOT_DIR/scripts/run_latency_probe.sh"

# 2) Phase 2 RNNT baselines (short sanity)
bash "$ROOT_DIR/scripts/run_phase2_baselines.sh"

# 3) Append summaries into docs (best-effort)
if [[ -f "$COREML_DIR/latency_sweep.md" ]]; then
  {
    printf '\n## Latency sweep results\n'
    echo
    cat "$COREML_DIR/latency_sweep.md"
  } >> "$PLAN_MD" || true
fi

# Note in training notes
if [[ -f "$COREML_DIR/latency_sweep.md" ]]; then
  {
    printf '\n- Ran latency sweep (modes=%s, chunks=%s). See exports/CoreMLTraces/latency_sweep.md\n' "$MODES_ENV" "$CHUNKS_ENV"
  } >> "$NOTES_MD" || true
fi

echo "Phase 3 report complete. Updated $PLAN_MD and $NOTES_MD"
