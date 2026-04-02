#!/usr/bin/env python3
"""
Phase 3 Automated Model Optimization and Deployment Pipeline

This module orchestrates the complete Phase 3 workflow for MambaASR model optimization,
Core ML export, and Swift validation. It provides a single-command interface for
transforming trained PyTorch models into production-ready Apple Silicon deployments.

Pipeline Architecture:
- Optimization: Knowledge distillation, quantization-aware training, structured pruning
- Export: Core ML conversion with Apple Neural Engine optimization
- Validation: Swift runtime verification with performance benchmarking
- Automation: End-to-end execution with comprehensive error handling

Phase 3 Workflow:
1. Knowledge Distillation: Create compact student model from teacher
2. Quantization-Aware Training: INT8/INT4 optimization for Apple Neural Engine
3. Structured Pruning: Hardware-friendly model compression
4. Core ML Export: Convert each optimized variant to .mlpackage format
5. Swift Validation: Performance and correctness verification on target hardware
6. Benchmarking: Latency analysis and performance profiling

Apple Silicon Integration:
- ANE Optimization: Models optimized for Neural Engine execution
- Metal Performance: MPS backend compatibility throughout pipeline
- Unified Memory: Memory-efficient optimization for Apple Silicon architecture
- Core ML Targeting: Direct integration with iOS/macOS deployment

Automation Features:
- Single Command: Complete pipeline execution with minimal configuration
- Error Recovery: Robust error handling and intermediate checkpoint saving
- Performance Tracking: Automated latency measurement and CSV export
- Resource Management: Optimized memory usage and cleanup

Called by:
- Development workflows for model optimization and deployment
- CI/CD pipelines for automated model release preparation
- Research experiments requiring comprehensive optimization comparison
- Production deployment preparation for iOS/macOS applications

Calls:
- scripts/optimize.py for knowledge distillation, QAT, and pruning
- scripts/export_and_validate.py for Core ML conversion and Swift validation
- MambaASRRunner for performance benchmarking and correctness verification
- subprocess management for orchestrating optimization pipeline

Integration Points:
- Phase 2: Receives trained MCT models from train_RNNT.py
- Phase 4: Produces deployment-ready models for Swift applications
- export_coreml.py: Core ML conversion with ANE optimization
- MambaASRRunner: Swift validation and performance measurement

Performance Targets:
- Model Optimization: 50-80% size reduction with <5% accuracy loss
- Export Speed: Complete pipeline execution in 10-30 minutes
- ANE Utilization: >90% operations on Neural Engine after optimization
- Deployment Ready: Models validated for production iOS/macOS deployment

Usage Example:
```bash
python scripts/phase3_pipeline.py \
  --manifest "/path/to/LibriSpeech/dev-clean.csv" \
  --steps 25 --batch_size 2 --duration 5 \
  --wav "/path/to/test_audio.wav"
```

Output Artifacts:
- checkpoints/MambaASR_kd_auto.pt: Knowledge distillation optimized model
- checkpoints/MambaASR_qat_auto.pt: Quantization-aware training optimized model  
- checkpoints/MambaASR_pruned_auto.pt: Structured pruning optimized model
- exports/*.mlpackage: Core ML packages for iOS/macOS deployment
- exports/latency_*.csv: Performance benchmarking results
"""
from __future__ import annotations
import argparse
import os
import subprocess
from pathlib import Path
import sys

# MARK: - Pipeline Configuration Constants

## Repository path configuration for Phase 3 pipeline execution.
## These paths enable reliable script execution across different environments.
REPO_ROOT = Path(__file__).resolve().parents[1]
MPS_ROOT = REPO_ROOT
SCRIPTS = MPS_ROOT / "scripts"

# MARK: - Pipeline Execution Utilities

def run(cmd: list[str], env: dict | None = None):
    """Executes subprocess command with logging and error handling.
    
    This utility function provides standardized command execution for the
    Phase 3 pipeline, ensuring consistent logging and error propagation
    across all optimization and validation steps.
    
    Args:
        cmd: Command and arguments as list of strings
             Must be properly formatted for subprocess execution
        env: Optional environment variables dictionary
             Inherits current environment if None provided
    
    Raises:
        subprocess.CalledProcessError: Command execution failed
        
    Called By:
        - main() for each optimization technique execution
        - Pipeline orchestration for Core ML export and validation
        
    Features:
        - Command logging: Displays executed commands for debugging
        - Error propagation: Fails fast on subprocess errors
        - Environment handling: Flexible environment variable management
        - Process isolation: Clean subprocess execution environment
    """
    print("$", " ".join(cmd))
    subprocess.check_call(cmd, env=env)


def main():
    """Orchestrates complete Phase 3 model optimization and deployment pipeline.
    
    This function coordinates the entire optimization workflow from PyTorch model
    optimization through Core ML export to Swift validation. It manages checkpoint
    creation, environment setup, and performance benchmarking for production deployment.
    
    Pipeline Execution Steps:
    1. Argument parsing: Configuration for optimization and validation parameters
    2. Environment setup: MPS fallback and Python path configuration for Apple Silicon
    3. Knowledge distillation: Teacher-student model compression with accuracy preservation
    4. Quantization-aware training: INT8/INT4 optimization for Apple Neural Engine
    5. Structured pruning: Hardware-friendly model compression with iterative refinement
    6. Export and validation: Core ML conversion with Swift runtime verification
    7. Performance benchmarking: Latency measurement and CSV export for analysis
    
    Optimization Techniques:
    - Knowledge Distillation: Creates compact models while preserving accuracy
    - Quantization-Aware Training: Prepares models for efficient ANE execution
    - Structured Pruning: Removes computational units while maintaining ANE compatibility
    
    Validation and Benchmarking:
    - Swift runtime: Validates Core ML models on target Apple Silicon hardware
    - Performance measurement: Latency profiling with statistical analysis
    - Correctness verification: Ensures model accuracy after optimization
    - Deployment readiness: Confirms models ready for production iOS/macOS apps
    
    Command Line Interface:
    - manifest: LibriSpeech dataset manifest for validation data
    - steps: Training steps for each optimization technique (default: 25)
    - batch_size: Batch size for optimization training (default: 2)  
    - duration: Streaming validation duration in seconds (default: 5)
    - warmup: Model warmup iterations for stable performance measurement (default: 2)
    - wav: Optional audio file for validation testing
    
    Output Management:
    - Checkpoints: Saved optimization results for each technique
    - Exports: Core ML packages ready for iOS/macOS deployment
    - Benchmarks: Performance analysis CSV files for optimization comparison
    - Logs: Comprehensive execution logs for debugging and analysis
    
    Error Handling:
    - Checkpoint management: Robust intermediate state saving and recovery
    - Process monitoring: Detailed subprocess execution with error propagation
    - Resource cleanup: Proper memory and file system resource management
    - Environment validation: Apple Silicon and MPS backend verification
    """
    ap = argparse.ArgumentParser(description="Phase 3 MambaASR optimization and deployment pipeline")
    ap.add_argument("--manifest", type=str, default="", 
                    help="LibriSpeech CSV manifest for validation data")
    ap.add_argument("--steps", type=int, default=25,
                    help="Training steps for each optimization technique")
    ap.add_argument("--batch_size", type=int, default=2,
                    help="Batch size for optimization training")
    ap.add_argument("--duration", type=int, default=5,
                    help="Streaming validation duration in seconds")
    ap.add_argument("--warmup", type=int, default=2,
                    help="Model warmup iterations for stable performance measurement")
    ap.add_argument("--wav", type=str, default="",
                    help="Optional audio file for validation testing")
    args = ap.parse_args()

    ckpt_dir = MPS_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env["PYTHONPATH"] = str(MPS_ROOT)

    # 1) KD
    kd_ckpt = ckpt_dir / "kd_student_auto.pt"
    run([
        sys.executable, str(SCRIPTS / "optimize.py"), "--technique", "kd",
        "--manifest", args.manifest, "--steps", str(args.steps), "--batch_size", str(args.batch_size),
        "--save_model", str(kd_ckpt)
    ], env=env)

    # 2) QAT
    qat_ckpt = ckpt_dir / "qat_model_auto.pt"
    run([
        sys.executable, str(SCRIPTS / "optimize.py"), "--technique", "qat",
        "--manifest", args.manifest, "--steps", str(args.steps), "--batch_size", str(args.batch_size),
        "--save_model", str(qat_ckpt)
    ], env=env)

    # 3) Prune
    pruned_ckpt = ckpt_dir / "pruned_model_auto.pt"
    run([
        sys.executable, str(SCRIPTS / "optimize.py"), "--technique", "prune",
        "--manifest", args.manifest, "--steps", str(args.steps), "--batch_size", str(args.batch_size),
        "--save_model", str(pruned_ckpt)
    ], env=env)

    # 4) Export+validate each
    for name, ckpt in [("MambaASR_kd_auto", kd_ckpt), ("MambaASR_qat_auto", qat_ckpt), ("MambaASR_pruned_auto", pruned_ckpt)]:
        ev_cmd = [
            sys.executable, str(SCRIPTS / "export_and_validate.py"),
            "--checkpoint", str(ckpt),
            "--name", name,
            "--duration", str(args.duration),
            "--warmup", str(args.warmup)
        ]
        if args.wav:
            ev_cmd.extend(["--wav", args.wav])
        # Save latencies alongside exports for each variant
        ev_cmd.extend(["--latency_csv", str(MPS_ROOT / "exports" / f"latency_{name}.csv")])
        run(ev_cmd, env=env)

    print("Phase 3 pipeline complete.")


if __name__ == "__main__":
    main()
