"""
Mamba-ASR Phase 3 optimization and deployment scripts for Apple Silicon.

This package provides production-ready optimization and deployment tools for
Mamba-based speech recognition models. It implements comprehensive Phase 3
optimization pipelines specifically designed for Apple Neural Engine (ANE)
deployment and Core ML integration.

Package Overview:
- Model optimization: Knowledge distillation, QAT, and structured pruning
- Core ML export: ANE-optimized model conversion for iOS/macOS deployment
- Production pipeline: End-to-end workflow from trained models to deployment
- Apple Silicon focus: Hardware-specific optimizations throughout

Phase 3 Integration Strategy:
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌─────────────┐
│ Phase 2     │───▶│ optimize.py  │───▶│ export_coreml.py│───▶│ Phase 4     │
│ Training    │    │ Optimization │    │ Core ML Export  │    │ Swift App   │
│ (train_*.py)│    │ Pipeline     │    │ Pipeline        │    │ Integration │
└─────────────┘    └──────────────┘    └─────────────────┘    └─────────────┘

Module Dependencies:
scripts/
├── optimize.py          # Model optimization pipeline
├── export_coreml.py     # Core ML export and ANE targeting
└── __init__.py         # This module documentation

Core Functionality:

1. Model Optimization (optimize.py):
   - Knowledge Distillation: Teacher-student learning for compact models
   - Quantization-Aware Training: INT8/INT4 precision for ANE optimization
   - Structured Pruning: Hardware-friendly model compression
   - Apple Silicon Integration: MPS backend optimization throughout

2. Core ML Export (export_coreml.py):
   - Stateful Core ML: Efficient Mamba state management
   - ANE Targeting: Neural Engine optimization and validation
   - Streaming Support: Chunk-based inference for real-time applications
   - iOS/macOS Deployment: Universal .mlpackage generation

Apple Silicon Optimization Focus:
- Unified Memory Architecture: Efficient memory usage patterns
- Metal Performance Shaders: MPS backend integration
- Apple Neural Engine: ANE-specific operation mapping
- Core ML Integration: Seamless iOS/macOS deployment preparation

Production Workflow:
1. Input: Trained MCT models from Phase 2 (train_RNNT.py)
2. Optimization: Knowledge distillation, QAT, or pruning (optimize.py)
3. Validation: Accuracy and performance verification
4. Export: Core ML conversion with ANE optimization (export_coreml.py)
5. Deployment: iOS/macOS integration in Phase 4

Performance Targets:
- Model Size: 50-80% reduction from baseline models
- Accuracy: <5% WER degradation from full-precision models
- ANE Utilization: >90% operations running on Neural Engine
- Inference Speed: <10ms latency for 10-second audio chunks

Usage Examples:

    # Knowledge Distillation Workflow
    from scripts.optimize import knowledge_distillation
    from scripts.export_coreml import export_to_coreml
    
    # Optimize large model to compact model
    student_model = knowledge_distillation(
        student_model=compact_mct,
        teacher_model=large_mct,
        train_dataloader=train_loader,
        val_dataloader=val_loader
    )
    
    # Export optimized model to Core ML
    export_to_coreml(
        pytorch_model=student_model,
        output_path="OptimizedMambaASR.mlpackage"
    )

    # Command Line Usage
    python scripts/optimize.py --technique kd --teacher large.pth --student compact.pth
    python scripts/export_coreml.py --model optimized.pth --output MambaASR.mlpackage

Integration Points:
- Input Pipeline: Trained models from train_CTC.py and train_RNNT.py
- Data Sources: LibriSpeech datasets via datasets/librispeech_csv.py
- Model Architectures: ConMambaCTC, MCTModel, TransformerASR
- Validation Tools: benchmarks/bench_mps.py for performance verification
- Output Target: Phase 4 Swift application integration

Apple Silicon Considerations:
- MPS Backend: GPU acceleration during optimization and validation
- Unified Memory: Efficient memory usage for large model processing
- Native ARM64: Optimized compilation and execution
- Core ML Tools: Apple's official conversion and optimization pipeline

Error Handling Strategy:
- Graceful Degradation: Continue processing with suboptimal configurations
- Validation Checks: Comprehensive accuracy and performance verification
- Fallback Options: Alternative optimization paths for edge cases
- Clear Diagnostics: Detailed error messages for debugging

Performance Monitoring:
- Training Metrics: Loss convergence and accuracy tracking
- Memory Usage: Unified memory pressure monitoring
- ANE Utilization: Neural Engine execution verification
- Inference Latency: Real-time performance measurement

Future Roadmap:
- MLX Integration: Apple's native ML framework support
- Advanced Quantization: INT4 and mixed-precision strategies
- Distributed Optimization: Multi-device training acceleration
- Real-time Optimization: Live model adaptation techniques

References:
- Core ML Documentation: Apple's machine learning framework
- Metal Performance Shaders: GPU acceleration on Apple Silicon
- Apple Neural Engine: On-device inference acceleration
- Knowledge Distillation: Hinton et al. model compression techniques
"""

from .optimize import (
    OptimizationConstants,
    knowledge_distillation,
    quantization_aware_training,
    structured_pruning
)

from .export_coreml import (
    CoreMLConstants,
    export_to_coreml
)

__all__ = [
    # Optimization Functions
    "knowledge_distillation",
    "quantization_aware_training", 
    "structured_pruning",
    "OptimizationConstants",
    
    # Core ML Export Functions
    "export_to_coreml",
    "CoreMLConstants",
]

# Version and Metadata
__version__ = "1.0.0"
__author__ = "Mamba-ASR Team"
__description__ = "Phase 3 optimization and deployment pipeline for Apple Silicon"