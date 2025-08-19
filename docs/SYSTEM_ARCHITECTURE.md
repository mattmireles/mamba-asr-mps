# MambaASR System Architecture Documentation

## Overview

This document provides comprehensive system architecture documentation for MambaASR, designed with AI-first development principles. Every component, integration point, and data flow is documented to enable seamless development and maintenance by AI systems.

## System Architecture Hierarchy

### Level 1: Core System Components

```
MambaASR System
├── Training Pipeline (modules/, train_*.py)
├── Model Architecture (modules/mct/, modules/mamba/)
├── Evaluation Pipeline (utils/, scripts/compute_*.py)
├── Export Pipeline (scripts/export_*.py)
├── Inference Runtime (swift/MambaASRRunner/)
└── Configuration Management (config/)
```

### Level 2: Integration Flow

```
Data → Training → Optimization → Export → Deployment → Evaluation
  ↓        ↓           ↓          ↓         ↓           ↓
CSV → MCT Model → KD/QAT/Prune → Core ML → Swift App → WER/CER
```

## Component Integration Matrix

### Training Pipeline Integration

| Component | File Location | Integrates With | Data Flow | Purpose |
|-----------|---------------|----------------|-----------|---------|
| `train_RNNT.py` | `/` | `modules/mct/`, `utils/`, `datasets/` | CSV → Model → Checkpoints | RNN-T training orchestration |
| `MCTModel` | `modules/mct/mct_model.py` | `modules/mamba/`, `modules/mct/joiner.py` | Audio → Encoder → Predictor → Joiner → Logits | Core architecture |
| `RNNTLoss` | `modules/rnnt_loss_mps.py` | `train_RNNT.py`, backend selection | Logits + Targets → Loss + Gradients | Loss computation with MPS optimization |
| `LibriSpeechDataset` | `datasets/librispeech_csv.py` | `train_RNNT.py`, `utils/tokenizer.py` | CSV → Audio Features + Token Sequences | Data loading and preprocessing |

### Model Architecture Integration

| Component | File Location | Input | Output | Dependencies |
|-----------|---------------|-------|--------|--------------|
| `MambaEncoder` | `modules/mct/encoder_mamba.py` | `(B, T, 80)` mel features | `(B, T', d_model)` encoded features | `modules/mamba/selective_scan_interface.py` |
| `SelectiveScan` | `modules/mamba/selective_scan_interface.py` | SSM parameters | Hidden state transitions | Pure PyTorch operations (MPS compatible) |
| `CNNFrontend` | `modules/mct/frontend_cnn.py` | `(B, T, 80)` mel features | `(B, T', d_model)` processed features | Standard PyTorch Conv1d layers |
| `RNNTPredictor` | `modules/mct/predictor.py` | Previous tokens | Predictor hidden states | LSTM-based language modeling |
| `Joiner` | `modules/mct/joiner.py` | Encoder + Predictor states | Final logits over vocabulary | Feedforward fusion network |

### Apple Silicon Optimization Integration

| Optimization | File Location | Target Component | Integration Point | Performance Impact |
|--------------|---------------|------------------|-------------------|-------------------|
| MPS Backend | `config/apple_silicon_config.py` | All PyTorch operations | Device selection in training scripts | 2-5x speedup over CPU |
| RNN-T Loss Fallback | `modules/rnnt_loss_mps.py` | Training pipeline | Loss computation | Maintains training stability |
| Memory Management | `config/apple_silicon_config.py` | Model training | Batch size and memory allocation | Prevents system swapping |
| Core ML Export | `scripts/export_coreml.py` | Trained models | PyTorch → Core ML conversion | ANE deployment optimization |

### Export and Deployment Pipeline

| Stage | Primary Script | Input | Output | Integration Points |
|-------|----------------|-------|--------|--------------------|
| Model Export | `scripts/export_coreml.py` | PyTorch checkpoint | `.mlpackage` file | `modules/mct/mct_model.py` for model loading |
| Compilation | `scripts/export_and_validate.py` | `.mlpackage` | `.mlmodelc` file | Core ML compiler integration |
| Swift Validation | `swift/MambaASRRunner/` | `.mlmodelc` + audio | Transcripts + latency metrics | Core ML runtime integration |
| Performance Analysis | `scripts/summarize_latency_csv.py` | Latency CSV files | Performance reports | Statistical analysis and reporting |

### Evaluation and Metrics Integration

| Evaluation Component | File Location | Input Sources | Output Format | Integration Points |
|---------------------|---------------|---------------|---------------|-------------------|
| WER/CER Computation | `scripts/compute_wer_cer.py` | Swift transcript outputs | Markdown reports | `utils/metrics.py` for core algorithms |
| Batch Evaluation | `scripts/eval_batch.sh` | Test audio directory | Transcript files + accuracy reports | `swift/MambaASRRunner/` for inference |
| Latency Analysis | `scripts/run_latency_probe.sh` | Core ML models | Performance summaries | Multiple compute configurations |
| Statistical Reporting | `scripts/summarize_latency_csv.py` | Raw measurement data | Statistical summaries | Percentile computation and report generation |

## Data Flow Architecture

### Training Data Flow

```
1. LibriSpeech CSV Manifest
   ↓ (librispeech_csv.py)
2. Audio Files + Transcripts
   ↓ (torchaudio loading + mel extraction)
3. Mel Features (B, T, 80) + Token Sequences
   ↓ (MCTModel forward pass)
4. Encoder Features + Predictor States
   ↓ (Joiner network)
5. Logits over Vocabulary
   ↓ (RNN-T Loss computation)
6. Loss + Gradients
   ↓ (Optimizer step)
7. Updated Model Parameters
```

### Inference Data Flow

```
1. Audio Input (16kHz mono WAV)
   ↓ (Core ML preprocessing)
2. Mel Features (1, T, 80)
   ↓ (Core ML model inference)
3. Logits over Vocabulary (1, T', V)
   ↓ (Swift decoding algorithms)
4. Token Sequences
   ↓ (Character to text conversion)
5. Final Transcript Text
```

### Optimization Pipeline Data Flow

```
1. Trained PyTorch Checkpoint
   ↓ (scripts/optimize.py)
2. Knowledge Distillation / QAT / Pruning
   ↓ (Optimized checkpoint)
3. Core ML Export
   ↓ (scripts/export_coreml.py)
4. .mlpackage File
   ↓ (Core ML compilation)
5. .mlmodelc File
   ↓ (Swift validation)
6. Performance Metrics + Deployment Package
```

## Cross-File Dependencies

### Critical Dependency Chains

#### Training → Inference Chain
```
train_RNNT.py → MCTModel → Core ML Export → Swift Runtime
       ↓              ↓            ↓             ↓
Configuration  Model State   .mlpackage   Transcripts
Constants      Management    Generation   + Metrics
```

#### Evaluation Chain
```
Swift Transcripts → WER/CER Computation → Performance Reports
        ↓                    ↓                    ↓
   stdout capture     Text normalization    Markdown output
```

#### Configuration Chain
```
Environment Variables → Configuration Classes → Component Initialization
         ↓                      ↓                        ↓
   Runtime overrides    Centralized constants    Consistent behavior
```

### Import Dependencies Graph

```
train_RNNT.py
├── modules.mct.mct_model (MCTModel, MCTConfig)
├── modules.rnnt_loss_mps (rnnt_loss_mps)
├── datasets.librispeech_csv (LibriSpeechCSVDataset)
├── utils.tokenizer (CharTokenizer)
├── utils.metrics (wer)
└── config.core_config (MambaASRConfig)

modules/mct/mct_model.py
├── modules.mct.encoder_mamba (MambaEncoder)
├── modules.mct.frontend_cnn (CNNFrontend)
├── modules.mct.predictor (RNNTPredictor)
└── modules.mct.joiner (Joiner)

modules/mamba/selective_scan_interface.py
├── torch (PyTorch operations)
├── torch.nn.functional (activation functions)
└── torch.autograd.profiler (performance profiling)

scripts/export_coreml.py
├── modules.mct.mct_model (model loading)
├── coremltools (conversion framework)
└── config.apple_silicon_config (ANE optimization)
```

## System State Management

### Training State Lifecycle

| State | File Location | Persistence | Restoration | Cleanup |
|-------|---------------|-------------|-------------|---------|
| Model Parameters | `train_RNNT.py` checkpoint saving | `.pt` files in `checkpoints/` | `torch.load()` with device mapping | Automatic via checkpoint rotation |
| Optimizer State | Training loop state management | Embedded in checkpoint files | Restored with model parameters | Cleared on training restart |
| Configuration | `config/` module initialization | Environment variables + defaults | Runtime detection and validation | Static lifecycle |
| Device State | Apple Silicon device management | Runtime hardware detection | Dynamic device selection | MPS cache management |

### Inference State Lifecycle

| State | Component | Initialization | Runtime Management | Cleanup |
|-------|-----------|----------------|-------------------|---------|
| Core ML Model | Swift MambaASRRunner | Model loading from `.mlmodelc` | Prediction state management | Automatic memory management |
| Audio Processing | Core ML preprocessing | Configuration from model metadata | Chunk-based processing | Automatic buffer management |
| Decoding State | Swift beam search implementation | Vocabulary and algorithm setup | Token sequence management | Cleared between utterances |

### Error State Handling

| Error Type | Detection Point | Recovery Strategy | Fallback Behavior | Documentation Location |
|------------|-----------------|-------------------|-------------------|----------------------|
| MPS Unavailable | Device detection in training | Automatic CPU fallback | Full training on CPU | `config/apple_silicon_config.py` |
| RNN-T Backend Failure | Loss computation | CPU gradient fallback | Maintained training progress | `modules/rnnt_loss_mps.py` |
| Core ML Export Error | Model conversion | Error reporting + guidance | Manual investigation required | `scripts/export_coreml.py` |
| Memory Pressure | Runtime monitoring | Batch size reduction + cache clearing | Continued operation with lower throughput | `config/apple_silicon_config.py` |

## Performance Integration Points

### Bottleneck Analysis

| Component | Performance Characteristic | Optimization Strategy | Measurement Location |
|-----------|---------------------------|----------------------|---------------------|
| Selective Scan | O(T*d_model) sequential computation | Phase 2: Custom Metal kernel | `benchmarks/bench_selective_scan.py` |
| RNN-T Loss | O(T*U) alignment matrix computation | MPS optimization + CPU fallback | `modules/rnnt_loss_mps.py` backend selection |
| Core ML Inference | ANE vs CPU vs GPU execution | Compute unit configuration | `scripts/run_latency_probe.sh` |
| Memory Bandwidth | Unified memory architecture utilization | Batch size optimization | `config/apple_silicon_config.py` |

### Optimization Pipeline Integration

```
Training Performance → Model Optimization → Deployment Performance
        ↓                     ↓                      ↓
  FPS measurement      KD/QAT/Pruning        Latency measurement
  Memory monitoring    Size reduction        ANE utilization
  Backend selection    Accuracy preservation  Real-time capability
```

## Configuration Integration Architecture

### Hierarchical Configuration System

```
Environment Variables (highest priority)
├── MAMBA_* (training parameters)
├── RNNT_* (loss computation)
├── PYTORCH_* (Apple Silicon MPS)
└── System defaults (lowest priority)
```

### Configuration Flow

```
Environment Detection → Configuration Classes → Component Initialization
         ↓                      ↓                       ↓
   Runtime validation    Centralized constants    Consistent behavior
   Override handling     Type safety              Error prevention
   Documentation         Cross-component access   Performance optimization
```

## AI-First Development Integration

### Documentation Strategy

Every component includes:
1. **Purpose Documentation**: What the component does and why it exists
2. **Integration Documentation**: How it connects to other components
3. **Data Flow Documentation**: Input/output specifications and transformations
4. **Error Handling Documentation**: Failure modes and recovery strategies
5. **Performance Documentation**: Characteristics and optimization opportunities

### Cross-Component Relationships

All components explicitly document:
- **Called By**: Which components invoke this functionality
- **Calls**: Which other components this depends on
- **Data Dependencies**: Input requirements and output guarantees
- **Configuration Dependencies**: Required configuration and environment setup
- **Platform Dependencies**: Apple Silicon optimizations and requirements

### Development Workflow Integration

```
1. Code Modification
   ↓
2. Documentation Update (automatic via AI-first principles)
   ↓
3. Configuration Validation (centralized config system)
   ↓
4. Testing (comprehensive component and integration tests)
   ↓
5. Performance Validation (automated benchmarking)
   ↓
6. Deployment (Core ML export and Swift integration)
```

This architecture enables AI systems to understand, modify, and extend the MambaASR system efficiently by providing comprehensive context for every component and integration point.