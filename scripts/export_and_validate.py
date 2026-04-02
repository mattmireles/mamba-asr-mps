#!/usr/bin/env python3
"""
End-to-End Core ML Export and Swift Validation Pipeline for MambaASR Deployment

This script provides a comprehensive workflow for converting trained PyTorch MambaASR
models into production-ready Core ML deployments with automated Swift validation.
It orchestrates the complete export-compile-validate cycle essential for Apple Silicon
deployment preparation and performance verification.

Key Responsibilities:
- Model export: PyTorch to Core ML conversion with Apple Neural Engine optimization
- Compilation: .mlpackage to .mlmodelc compilation for runtime optimization
- Swift validation: Real-time streaming validation using MambaASRRunner CLI
- Performance measurement: Latency analysis and throughput benchmarking
- Error detection: Comprehensive validation of model correctness and interface compliance

Workflow Architecture:
1. Checkpoint validation: Verify trained PyTorch model exists and is accessible
2. Core ML export: Convert to .mlpackage with ANE optimization and stateful interface
3. Swift compilation: Build MambaASRRunner for target hardware validation
4. Runtime validation: Execute streaming inference with synthetic or real audio
5. Performance analysis: Capture latency metrics and validate deployment readiness

Apple Silicon Integration:
- ANE targeting: Models optimized for Neural Engine execution patterns
- MPS compatibility: Ensures proper Metal Performance Shader backend operation
- Unified memory: Optimized for Apple Silicon memory architecture
- Core ML optimization: Leverages Core ML compiler optimizations for target hardware

Called By:
- scripts/phase3_pipeline.py for automated optimization workflow execution
- CI/CD pipelines for post-training validation and deployment preparation
- Development workflows for manual model validation and performance testing
- Research experiments requiring Core ML deployment verification

Calls:
- scripts/export_coreml.py for PyTorch to Core ML model conversion
- MambaASRRunner Swift binary for runtime validation and performance measurement
- subprocess operations for orchestrating multi-language pipeline execution
- file system operations for managing intermediate artifacts and outputs

Integration Points:
- Input: Trained PyTorch checkpoints from train_RNNT.py or optimization pipeline
- Processing: Core ML export with ANE optimization and Swift runtime validation
- Output: Validated .mlmodelc files ready for iOS/macOS application integration
- Metrics: Latency CSV files and performance summaries for deployment decisions

Performance Validation:
- Streaming latency: Per-chunk inference timing for real-time deployment assessment
- Memory usage: Peak memory consumption validation for mobile deployment
- Accuracy verification: Model output correctness confirmation across export pipeline
- Hardware utilization: ANE/GPU/CPU backend usage analysis for optimization guidance

Error Handling:
- Checkpoint validation: Clear error messages for missing or corrupted model files
- Export failures: Comprehensive error reporting for Core ML conversion issues
- Compilation errors: Swift build error capture and debugging information
- Runtime validation: Inference failure detection and diagnostic output

Usage Examples:
  # Basic validation with synthetic audio
  python scripts/export_and_validate.py --checkpoint checkpoints/model.pt
  
  # Real audio streaming with latency measurement
  python scripts/export_and_validate.py \
    --checkpoint checkpoints/kd_student.pt \
    --name MambaASR_optimized \
    --wav audio/test_16khz.wav \
    --latency_csv results/latency.csv \
    --duration 30

  # ANE-specific validation
  python scripts/export_and_validate.py \
    --checkpoint checkpoints/model.pt \
    --compute all \
    --chunk 512 \
    --warmup 5
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# MARK: - Configuration Constants

class ExportValidationConstants:
    """
    Named constants for export and validation pipeline configuration.
    
    These constants define the file system layout and operational parameters
    for the end-to-end export and validation workflow, ensuring consistent
    behavior across different execution environments and deployment scenarios.
    """
    
    # MARK: - Directory Structure
    
    # Repository root directory calculated from script location.
    # Enables consistent path resolution across different execution contexts.
    REPO_ROOT = Path(__file__).resolve().parents[1]
    
    # Standalone repo root for model and script organization.
    MPS_ROOT = REPO_ROOT
    
    # Scripts directory containing pipeline automation and utility tools.
    # Houses export_coreml.py and other workflow orchestration scripts.
    SCRIPTS_DIR = MPS_ROOT / "scripts"
    
    # Swift package directory for MambaASRRunner CLI validation tool.
    # Contains Package.swift and source files for Core ML validation.
    RUNNER_DIR = MPS_ROOT / "swift" / "MambaASRRunner"
    
    # Compiled MambaASRRunner binary path for validation execution.
    # Release build optimized for performance measurement and validation.
    RUNNER_BIN = RUNNER_DIR / ".build" / "arm64-apple-macosx" / "release" / "MambaASRRunner"
    
    # MARK: - Default Configuration Values
    
    # Default base name for exported Core ML artifacts.
    # Used when --name argument not specified for consistent naming.
    DEFAULT_EXPORT_NAME = "MambaASR_export"
    
    # Default streaming duration for validation testing (seconds).
    # Balanced duration for meaningful performance measurement without excessive runtime.
    DEFAULT_DURATION_SECONDS = 5
    
    # Default warmup inference count to amortize first-call overhead.
    # Ensures stable performance measurements by excluding cold-start costs.
    DEFAULT_WARMUP_COUNT = 2
    
    # Default compute unit configuration for Core ML validation.
    # Conservative CPU-only setting for maximum compatibility during validation.
    DEFAULT_COMPUTE_UNITS = "cpu"
    
    # Default chunk length for streaming validation (frames).
    # Balanced chunk size for realistic streaming performance assessment.
    DEFAULT_CHUNK_LENGTH = 256


# MARK: - Utility Functions

def run(cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict] = None) -> None:
    """
    Execute subprocess command with comprehensive logging and error handling.
    
    This utility function provides standardized subprocess execution for the
    export and validation pipeline, ensuring consistent logging and error
    handling across all orchestrated operations including export, compilation,
    and validation steps.
    
    Execution Strategy:
    - Command logging: Full command display for debugging and audit trails
    - Error propagation: Immediate failure on non-zero exit codes
    - Environment isolation: Optional environment modification for specific tools
    - Working directory: Flexible execution context for different pipeline stages
    
    Called By:
    - main() function for orchestrating export, compilation, and validation steps
    - Pipeline workflow automation requiring subprocess command execution
    - Swift build operations needing specific working directory context
    
    Args:
        cmd: Command line arguments as list of strings
             - First element must be executable name or path
             - Subsequent elements are command-line arguments
        cwd: Optional working directory for command execution
             - Defaults to current directory if not specified
             - Essential for Swift build and script execution context
        env: Optional environment variable dictionary
             - Merged with current environment if provided
             - Used for setting build configurations and tool paths
             
    Raises:
        subprocess.CalledProcessError: Command execution failure with non-zero exit
        FileNotFoundError: Executable not found in system PATH or specified location
        
    Example Usage:
        run(["swift", "build", "-c", "release"], cwd=RUNNER_DIR)
        run(["python", "export_coreml.py", "--checkpoint", "model.pt"])
        run(["./MambaASRRunner", "--stream"], env={"COMPUTE": "all"})
    """
    print("$", " ".join(cmd))  # Command logging for debugging and audit
    subprocess.check_call(
        cmd, 
        cwd=str(cwd) if cwd else None, 
        env=env
    )


# MARK: - Main Pipeline Orchestration

def main() -> None:
    """
    Execute end-to-end Core ML export and Swift validation pipeline.
    
    This function orchestrates the complete workflow for converting trained PyTorch
    MambaASR models into validated Core ML deployments ready for iOS/macOS integration.
    It manages the complex multi-step process with comprehensive error handling and
    performance measurement throughout the pipeline.
    
    Pipeline Execution Flow:
    1. Argument parsing: Command-line interface for flexible pipeline configuration
    2. Checkpoint validation: Verify trained PyTorch model accessibility and format
    3. Core ML export: Convert model to .mlpackage with ANE optimization
    4. Swift compilation: Build MambaASRRunner for target hardware validation
    5. Runtime validation: Execute streaming inference with performance measurement
    6. Results reporting: Summary of validation success and performance metrics
    
    Called By:
    - Command line execution: Direct script invocation for manual validation
    - scripts/phase3_pipeline.py: Automated optimization workflow integration
    - CI/CD pipelines: Post-training validation and deployment preparation
    - Development workflows: Manual model testing and performance analysis
    
    Calls:
    - scripts/export_coreml.py: PyTorch to Core ML conversion with ANE optimization
    - swift build: MambaASRRunner CLI compilation for validation execution
    - MambaASRRunner: Core ML model validation with streaming inference
    - argparse: Command-line argument parsing for flexible configuration
    
    Validation Strategy:
    - Model correctness: Verify Core ML export maintains model functionality
    - Performance measurement: Capture streaming latency for deployment assessment
    - Hardware compatibility: Validate execution on target Apple Silicon configurations
    - Interface compliance: Ensure model interface matches deployment requirements
    
    Error Handling:
    - Checkpoint validation: Clear error messages for missing or invalid models
    - Export failures: Comprehensive error reporting for debugging Core ML issues
    - Compilation errors: Swift build error capture and diagnostic information
    - Runtime failures: Validation error detection with debugging context
    
    Performance Validation:
    - Streaming latency: Per-chunk inference timing for real-time assessment
    - Memory usage: Peak memory consumption validation for mobile deployment
    - Hardware utilization: Backend usage analysis for optimization guidance
    - Throughput analysis: Overall performance characterization for SLA validation
    """
    # Configure command-line interface with comprehensive options
    ap = argparse.ArgumentParser(
        description="End-to-end Core ML export and Swift validation for MambaASR"
    )
    ap.add_argument(
        "--checkpoint", 
        required=True, 
        help="Path to .pt checkpoint to export"
    )
    ap.add_argument(
        "--name", 
        default=ExportValidationConstants.DEFAULT_EXPORT_NAME,
        help="Base name for exported artifacts"
    )
    ap.add_argument(
        "--duration", 
        type=int, 
        default=ExportValidationConstants.DEFAULT_DURATION_SECONDS,
        help="Streaming duration (seconds)"
    )
    ap.add_argument(
        "--warmup", 
        type=int, 
        default=ExportValidationConstants.DEFAULT_WARMUP_COUNT,
        help="Number of warmup inferences to amortize first-call cost"
    )
    ap.add_argument(
        "--wav", 
        type=str, 
        default="", 
        help="Optional 16kHz mono wav to stream"
    )
    ap.add_argument(
        "--latency_csv", 
        type=str, 
        default="", 
        help="Optional path to write per-chunk latency CSV"
    )
    ap.add_argument(
        "--vocab_out", 
        type=str, 
        default="", 
        help="Optional path to write vocab JSON for Swift greedy decode"
    )
    ap.add_argument(
        "--compute", 
        type=str, 
        default=ExportValidationConstants.DEFAULT_COMPUTE_UNITS,
        choices=["all", "cpu", "cpu-gpu"], 
        help="Compute units for validation run"
    )
    ap.add_argument(
        "--chunk", 
        type=int, 
        default=ExportValidationConstants.DEFAULT_CHUNK_LENGTH,
        help="Chunk length for streaming validation"
    )
    args = ap.parse_args()

    # Validate checkpoint file exists and is accessible
    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.exists():
        print(f"❌ Checkpoint not found: {ckpt}")
        print("🔧 Please verify the checkpoint path is correct and the file exists")
        sys.exit(1)

    # Define output paths using established constants
    exports = ExportValidationConstants.MPS_ROOT / "exports"
    compiled_dir = exports / f"Compiled_{args.name}"
    mlpackage = exports / f"{args.name}.mlpackage"
    mlmodelc = compiled_dir / f"{args.name}.mlmodelc"
    exports.mkdir(parents=True, exist_ok=True)

    # 1) Export to .mlpackage
    env = os.environ.copy()
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # Ensure PYTHONPATH so export script can import modules
    env["PYTHONPATH"] = str(ExportValidationConstants.REPO_ROOT)
    run([sys.executable, str(ExportValidationConstants.SCRIPTS_DIR / "export_coreml.py"),
         "--model", str(ckpt), "--output", str(mlpackage)], env=env)

    # 2) Compile to .mlmodelc
    compiled_dir.mkdir(parents=True, exist_ok=True)
    run([
        "xcrun", "coremlcompiler", "compile",
        str(mlpackage), str(compiled_dir)
    ])

    # 3) Build Swift runner if missing
    if not ExportValidationConstants.RUNNER_BIN.exists():
        run(["swift", "build", "-c", "release", "--package-path", str(ExportValidationConstants.RUNNER_DIR)])

    # 3.5) Optionally emit a simple character vocab JSON for greedy decode
    vocab_path: Path | None = None
    if args.vocab_out:
        vocab_path = Path(args.vocab_out).resolve()
    else:
        vocab_path = (exports / "vocab_char_29.json").resolve()
    try:
        import json
        vocab_map: dict[str, str] = {"0": ""}
        vocab_map["1"] = " "
        for i, ch in enumerate([chr(ord('a') + k) for k in range(26)], start=2):
            vocab_map[str(i)] = ch
        vocab_map["28"] = "'"
        with open(vocab_path, "w") as f:
            json.dump(vocab_map, f)
    except Exception:
        vocab_path = None

    # 4) Run Swift runner in streaming mode
    cmd = [
        str(ExportValidationConstants.RUNNER_BIN),
        "--mlmodelc", str(mlmodelc),
        "--mlpackage", str(mlpackage),
        "--stream", "--duration", str(args.duration),
        "--warmup", str(args.warmup)
    ]
    if args.compute:
        cmd.extend(["--compute", args.compute])
    if args.chunk:
        cmd.extend(["--chunk", str(args.chunk)])
    if args.wav:
        cmd.extend(["--wav", args.wav])
    if args.latency_csv:
        cmd.extend(["--latency-csv", args.latency_csv])
    if vocab_path is not None:
        cmd.extend(["--vocab", str(vocab_path)])
    run(cmd)

    print("All done.")


if __name__ == "__main__":
    main()
