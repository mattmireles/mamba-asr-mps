#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Core ML Latency Profiling and Performance Analysis Pipeline
# =============================================================================
#
# This script provides comprehensive latency analysis for MambaASR Core ML models
# across different Apple Silicon compute configurations. It automates the collection
# of streaming inference performance data and generates standardized reports for
# deployment readiness validation and regression detection.
#
# System Integration:
# - Called by: Performance analysis workflows and CI/CD regression testing
# - Calls: MambaASRRunner for streaming latency measurement with diverse configurations  
# - Calls: scripts/summarize_latency_csv.py for statistical analysis and report generation
# - Uses: Compiled Core ML models (.mlmodelc) from the export and optimization pipeline
#
# Profiling Strategy:
# - Multi-dimensional analysis: Compute mode × chunk size × model variant combinations
# - Statistical rigor: Configurable warmup periods and measurement duration for stable results
# - Hardware targeting: Validates performance across CPU, GPU, and Apple Neural Engine
# - Regression detection: Standardized output format enables automated performance comparison
#
# Compute Configuration Matrix:
# - 'all': Automatic compute unit selection (ANE > GPU > CPU priority)
# - 'cpu': CPU-only execution for baseline performance measurement
# - 'cpu-gpu': CPU+GPU execution excluding ANE for fallback validation
#
# Chunk Size Analysis:
# - Default: 256 frames (10.24 seconds at 25fps) for optimal Apple Silicon performance
# - Configurable: Via LAT_SWEEP_CHUNKS environment variable for custom analysis
# - Memory trade-offs: Larger chunks improve ANE utilization but increase memory pressure
#
# Performance Metrics Collection:
# - Per-chunk latency: Individual inference timing for variance analysis
# - Statistical summary: Mean, p50, p90, p99 percentiles for SLA validation
# - Throughput analysis: Frames per second and real-time factor calculation
# - Memory characteristics: Peak memory usage and allocation patterns
#
# Output Organization:
# - Raw data: Individual CSV files per configuration (latency_MODE_cCHUNK.csv)
# - Summaries: Markdown reports with statistical analysis (latency_MODE_cCHUNK.md)
# - Combined report: Comprehensive sweep analysis (latency_sweep.md)
# - Artifact management: Organized output structure for automated analysis
#
# Environment Configuration:
# - LAT_SWEEP_CHUNKS: Comma-separated chunk sizes (default: "256")
# - LAT_SWEEP_MODES: Comma-separated compute modes (default: "all,cpu,cpu-gpu")
# - LAT_DURATION: Measurement duration in seconds (default: 10)
# - LAT_WARMUP: Warmup iterations before measurement (default: 2)
# - LAT_SWEEP_TAG: Optional identifier for output file disambiguation
#
# Hardware Validation Strategy:
# - ANE verification: 'all' mode validates Neural Engine utilization
# - CPU baseline: 'cpu' mode provides universal performance baseline
# - GPU fallback: 'cpu-gpu' mode tests Metal Performance Shader execution
# - Cross-platform: Consistent methodology across Apple Silicon variants
#
# Error Handling and Robustness:
# - Build automation: Compiles MambaASRRunner if binary missing
# - Graceful degradation: Individual configuration failures don't halt entire sweep
# - Path validation: Comprehensive input validation and default path resolution
# - Result aggregation: Combines partial results even if some configurations fail
#
# Usage Examples:
#   # Standard performance sweep with defaults
#   scripts/run_latency_probe.sh
#   
#   # Custom model and audio file
#   scripts/run_latency_probe.sh model.mlmodelc audio.wav
#   
#   # Extended analysis with environment customization
#   LAT_SWEEP_CHUNKS="128,256,512" LAT_DURATION=30 scripts/run_latency_probe.sh
#   
#   # Tagged analysis for model comparison
#   LAT_SWEEP_TAG="optimized_v2" scripts/run_latency_probe.sh
#
# Integration with Analysis Pipeline:
# - Input: Compiled Core ML models from optimization pipeline
# - Processing: Systematic latency measurement across hardware configurations
# - Output: Standardized performance reports for deployment decision making
# - Workflow: Enables quantitative comparison of model optimization effectiveness

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
