#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
EXPORTS_DIR="$ROOT_DIR/exports"
mkdir -p "$EXPORTS_DIR"
COREML_DIR="$EXPORTS_DIR/CoreMLTraces"
mkdir -p "$COREML_DIR"

SANITY_FLAGS=(--sanity --epochs 1 --batch_size 2 --max_steps 60)

run_case() {
  local name="$1"; shift
  local json_out="$EXPORTS_DIR/phase2_${name}_60.summary.json"
  local csv_out="$EXPORTS_DIR/phase2_${name}_60.csv"
  PYTORCH_ENABLE_MPS_FALLBACK=1 \
  PYTHONPATH="$ROOT_DIR" \
  python "$ROOT_DIR/train_RNNT.py" \
    "${SANITY_FLAGS[@]}" \
    --log_json "$json_out" --log_csv "$csv_out" \
    "$@"
}

run_case mps_native --rnnt_impl mps_native
run_case auto       --rnnt_impl auto
run_case cpu_grad   --force_cpu_grad
run_case ctc        --rnnt_impl ctc

echo "Phase 2 baseline runs complete. See $EXPORTS_DIR for CSV/JSON outputs." 

# If latency CSV exists from Swift runner, summarize it for the docs
LATCSV="$COREML_DIR/latency_probe.csv"
if [[ -f "$LATCSV" ]]; then
  python "$ROOT_DIR/scripts/summarize_latency_csv.py" \
    --csv "$LATCSV" \
    --out "$COREML_DIR/latency_summary.md" || true
  echo "Latency summary written to $COREML_DIR/latency_summary.md"
fi

# Auto-embed latest latency summary into implementation plan (optional best-effort)
PLAN_MD="$ROOT_DIR/README/implementation-plan-v2.md"
if [[ -f "$COREML_DIR/latency_summary.md" && -f "$PLAN_MD" ]]; then
  printf '\nAppending latest latency summary into plan...\n'
  {
    printf '\nLatest streaming latency summary:\n';
    echo;
    echo '```text';
    cat "$COREML_DIR/latency_summary.md";
    echo '```';
  } >> "$PLAN_MD" || true
fi
