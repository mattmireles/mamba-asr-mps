#!/usr/bin/env bash
set -euo pipefail
# Orchestrates Core ML runner latency probes and summaries.
# Produces per-combination CSV/MD and a combined sweep summary.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUNNER="$ROOT_DIR/swift/MambaASRRunner/.build/arm64-apple-macosx/release/MambaASRRunner"
MLMODELC_DEFAULT="$ROOT_DIR/exports/Compiled_fp16_w8/MambaASR_fp16_w8.mlmodelc"
WAV_DEFAULT="$ROOT_DIR/exports/tts_real_long_16k.wav"
COREML_DIR="$ROOT_DIR/exports/CoreMLTraces"
mkdir -p "$COREML_DIR"

# Ensure runner exists; if not, build it in release.
if [[ ! -x "$RUNNER" ]]; then
  echo "Runner not found at $RUNNER. Building in release..."
  (cd "$ROOT_DIR/swift/MambaASRRunner" && swift build -c release -Xswiftc -O)
fi

MLMODELC_PATH="${1:-$MLMODELC_DEFAULT}"
WAV_PATH="${2:-$WAV_DEFAULT}"

# Compute modes and chunk sizes to sweep
COMPUTE_MODES=(all cpu cpu-gpu)
CHUNKS=(256)

# Allow override via env
if [[ -n "${LAT_SWEEP_CHUNKS:-}" ]]; then
  IFS=',' read -r -a CHUNKS <<<"$LAT_SWEEP_CHUNKS"
fi
if [[ -n "${LAT_SWEEP_MODES:-}" ]]; then
  IFS=',' read -r -a COMPUTE_MODES <<<"$LAT_SWEEP_MODES"
fi

# Duration/warmup defaults
DURATION=${LAT_DURATION:-10}
WARMUP=${LAT_WARMUP:-2}

# Optional tag to distinguish outputs (e.g., model name)
SWEEP_TAG=${LAT_SWEEP_TAG:-}
SUFFIX="${SWEEP_TAG:+_${SWEEP_TAG}}"

COMBINED_MD="$COREML_DIR/latency_sweep${SUFFIX}.md"
: > "$COMBINED_MD"
echo "# Latency sweep (compute modes and chunk sizes)" >> "$COMBINED_MD"
echo "Model: $MLMODELC_PATH" >> "$COMBINED_MD"
echo "Modes: ${COMPUTE_MODES[*]} | Chunks: ${CHUNKS[*]} | Duration: $DURATION | Warmup: $WARMUP" >> "$COMBINED_MD"
echo "" >> "$COMBINED_MD"

for mode in "${COMPUTE_MODES[@]}"; do
  for chunk in "${CHUNKS[@]}"; do
    tag="${mode}_c${chunk}"
    csv_out="$COREML_DIR/latency_${tag}${SUFFIX}.csv"
    md_out="$COREML_DIR/latency_${tag}${SUFFIX}.md"
    echo "Running: compute=$mode chunk=$chunk"
    "$RUNNER" \
      --mlmodelc "$MLMODELC_PATH" \
      --stream --duration "$DURATION" --warmup "$WARMUP" \
      --compute "$mode" --chunk "$chunk" \
      --wav "$WAV_PATH" \
      --latency-csv "$csv_out" || true
    if [[ -f "$csv_out" ]]; then
      python "$ROOT_DIR/scripts/summarize_latency_csv.py" --csv "$csv_out" --out "$md_out"
      echo "## $tag" >> "$COMBINED_MD"
      echo "" >> "$COMBINED_MD"
      cat "$md_out" >> "$COMBINED_MD"
      echo "" >> "$COMBINED_MD"
    else
      echo "(no CSV produced for $tag)" >> "$COMBINED_MD"
    fi
  done
done

echo "Wrote combined sweep to $COMBINED_MD"
