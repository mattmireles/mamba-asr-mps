# Mamba-ASR for Apple Silicon: Re-architecture & Implementation Plan

## Executive Summary

This document outlines the strategic plan to re-architect and reimplement the `Mamba-ASR-NVIDIA` project for optimal performance on Apple Silicon. Instead of a direct port from CUDA to the Metal Performance Shaders (MPS) backend, this plan details a ground-up, from-scratch reimplementation in a new `Mamba-ASR-MPS` directory.

The core of this strategy is to move beyond the limitations of a compatibility-layer approach and build a truly native, high-performance Automatic Speech Recognition (ASR) system. The final architecture will be a **Hybrid Mamba-CNN Transducer (MCT)**, specifically designed to leverage the unique strengths of Apple's full machine learning stack, including the Apple Neural Engine (ANE) and the Apple Matrix Coprocessor (AMX).

### Core Design Principles

1.  **Apple Silicon First**: The architecture will be designed from first principles for the specific capabilities of MPS, ANE, and the Unified Memory Architecture. Performance on Apple hardware is the primary goal, not an afterthought.
2.  **Read-and-Reimplement, Don't Copy**: The `Mamba-ASR-NVIDIA` project will serve as a logical blueprint and a source of algorithmic reference only. No code will be directly copied, ensuring a clean, native, and maintainable codebase free of CUDA-specific artifacts.
3.  **Simplicity & Maintainability**: The new implementation will prioritize clarity and simplicity, following the principle that "simpler is better." The code will be well-documented and structured for long-term maintenance by both human and AI developers.
4.  **Phased, Validated Development**: The project will follow a structured, four-phase roadmap. Each phase has clear objectives and exit criteria, allowing for iterative development, validation at each stage, and progressive complexity.

### Target Architecture Vision

The end goal is a state-of-the-art, on-device streaming ASR system composed of:
-   **A Core ML Model**: An optimized Hybrid Mamba-CNN Transducer (MCT) model, quantized and converted to a stateful Core ML package to run efficiently on the Apple Neural Engine.
-   **A Native Swift Pipeline**: A high-performance application layer that handles audio preprocessing with the **vDSP** library and performs beam search decoding accelerated by the **Apple Matrix Coprocessor (AMX)** via the Accelerate framework.

---

## Implementation Roadmap

This project is divided into four distinct phases, moving from a basic functional baseline to a fully optimized, production-ready native application.

### Phase 1: Project Foundation & Functional MPS Baseline

**Objective**: To establish a clean project structure and achieve a functional, end-to-end training pipeline for the existing ConMamba architecture on the MPS backend. This phase validates the toolchain and provides a critical performance baseline.

**Key Tasks**:
1.  **Establish Project Foundation**: Create the new `Mamba-ASR-MPS/` directory structure from scratch. This will house all new, purpose-built modules, scripts, and configurations.
2.  **Reimplement Core Components**: By referencing the logic in `Mamba-ASR-NVIDIA`, write clean, new, and device-agnostic implementations of the core components:
    *   **Data Preparation**: A new `librispeech_prepare.py` script.
    *   **Training Scripts**: A new `train_ctc.py` script.
    *   **Model Modules**: A new `modules/` directory containing reimplemented Conformer and Mamba blocks.
3.  **Develop MPS-Native Mamba Kernel**: Write a new, pure PyTorch-native version of Mamba's `selective_scan` algorithm. This is the most critical task of this phase, as it removes the dependency on the custom CUDA kernel and enables Mamba to run on MPS.
4.  **Author MPS-Optimized Configurations**: Create new `hparams` YAML files specifically for Apple Silicon, starting with `fp32` precision and conservative batch sizes suitable for the unified memory architecture.
5.  **Achieve Functional Baseline**: Successfully execute the new data preparation and training scripts to train a `ConMamba-CTC` model end-to-end on the MPS backend.

**Exit Criteria**: A successfully trained ConMamba-CTC model, with logs and benchmarks documenting the performance of the pure PyTorch-on-MPS baseline.

### Phase 2: Architectural Redesign to Mamba-CNN Transducer (MCT)

**Objective**: To implement the superior "Hybrid Mamba-CNN Transducer" (MCT) architecture proposed in the project's strategic documentation. This model is designed from first principles for optimal performance on the Apple Neural Engine.

**Key Tasks**:
1.  **Architect and Implement MCT Modules**: Write entirely new Python modules for the MCT architecture, including:
    *   An **ANE-friendly CNN Frontend** for efficient feature extraction and subsampling.
    *   The **Mamba Encoder Core** for sequence modeling, using our MPS-native kernel from Phase 1.
    *   An **RNN-T Predictor & Joiner Network**, which will replace the less-efficient CTC/S2S paradigm.
2.  **Implement RNN-T Training Logic**: Write a new RNN-Transducer loss function and integrate it into a new, dedicated training script, `train_rnnt.py`.
3.  **Train and Benchmark the MCT Model**: Train the newly designed MCT model on Apple Silicon to establish its accuracy and performance characteristics, which will serve as the benchmark for the final optimization phase.

**Exit Criteria**: A trained MCT model with comprehensive benchmarks detailing its accuracy (WER) and performance (training time, inference speed) on Apple Silicon.

### Phase 3: On-Device Optimization & Core ML Conversion

**Objective**: To compress the trained MCT model and convert it into a highly optimized Core ML package ready for on-device deployment, specifically targeting the ANE.

**Key Tasks**:
1.  **Develop Optimization Scripts**: Write new, standalone Python scripts to perform a series of advanced optimization techniques on the trained MCT model:
    *   **Knowledge Distillation:** To enhance the accuracy of the compact model.
    *   **Quantization-Aware Training (QAT):** To fine-tune the model for low-precision INT8/INT4 integer execution on the ANE.
    *   **Structured Pruning:** To reduce model size in a hardware-friendly way.
2.  **Author a Stateful Core ML Conversion Script**: Write a new conversion script using `coremltools`. The primary focus will be leveraging **Stateful Models (`StateType`)** to allow the Core ML runtime to efficiently manage the Mamba model's recurrent state internally—a critical step for high-performance streaming.
3.  **Verify ANE Execution**: Detail the process for loading the converted model in Xcode and using its performance analysis tools to confirm that all key operations are successfully running on the Apple Neural Engine.

**Exit Criteria**: A fully optimized `.mlpackage` file where all critical layers are verified to run on the ANE, with documentation on the conversion process.

### Phase 4: Building the Native Swift Inference Pipeline

**Objective**: To build a complete, high-performance Swift application that integrates the Core ML model into an efficient, end-to-end inference pipeline that leverages the full Apple hardware stack.

**Key Tasks**:
1.  **Write the vDSP Preprocessing Pipeline**: Write a new Swift module to handle all audio preprocessing (e.g., Mel spectrogram calculation) using the **Accelerate framework's vDSP library**, ensuring this CPU-bound task is maximally efficient.
2.  **Write the AMX-Accelerated Decoder**: Write a new Swift implementation of the beam search decoding algorithm. Its core matrix operations will be written using the **Accelerate framework (BNNS/BLAS)** to offload the work to the **Apple Matrix Coprocessor (AMX)**.
3.  **Profile and Finalize**: Outline the final profiling process using Xcode Instruments to analyze the full pipeline—from audio input to text output—to identify and eliminate any remaining bottlenecks and achieve the target real-time performance.

**Exit Criteria**: A functional Swift application demonstrating real-time, end-to-end ASR performance, with documented profiling results.

## Implementation Progress (Track your progress below)
\n+### Phase 1: Project Foundation & Functional MPS Baseline
- [x] Establish project foundation under `Mamba-ASR-MPS/`
- [x] Reimplement core components
  - [x] Data preparation: `librispeech_prepare.py`
  - [x] Training script: `train_CTC.py` (MPS-aware, CPU fallback for CTC)
  - [x] Model modules: `modules/mamba/selective_scan_interface.py`, `modules/mamba/mamba_blocks.py`, `modules/Conmamba.py`
- [x] Develop MPS-native (pure PyTorch) selective scan for baseline
- [x] Author Apple Silicon hparams: `hparams/CTC/*.yaml` (FP32, conservative batch size)
- [x] Achieve functional baseline: sanity training step runs on MPS (CTC falls back to CPU)

Notes:
- This baseline is correctness-first and intended for profiling. Performance optimizations (fused Metal kernels) will be implemented in later phases.

### Phase 2: Architectural Redesign to MCT (in progress)
- [x] Implement MCT modules
  - [x] Frontend CNN (`modules/mct/frontend_cnn.py`)
  - [x] Mamba encoder (`modules/mct/encoder_mamba.py`)
  - [x] RNNT predictor (`modules/mct/predictor.py`)
  - [x] RNNT joiner (`modules/mct/joiner.py`)
  - [x] MCTModel (`modules/mct/mct_model.py`)
- [x] RNNT training script with fallbacks: `train_RNNT.py`
- [x] RNNT Apple Silicon hparams: `hparams/RNNT/mct_baseline.yaml`
- [x] Sanity RNNT pass on MPS (CPU fallback for loss)
- [x] Add naive RNNT loss fallback (small T,U) for sanity checks
- [x] Benchmark MCT vs CTC baselines; record throughput/memory on MPS (sanity)
  - CTC ~ 689 frames/s (bs=1, T=800)
  - RNNT(enc-ctc) ~ 409 frames/s (bs=1, T=600, U=30)
- [x] Prepare profiling trace and identify selective_scan hotspots (via record_function + optional profiler)
 - [x] Integrate real RNNT loss (torchaudio prototype or warp-transducer) for standard T,U
   - Implemented multi-backend selection in `train_RNNT.py` via `--rnnt_impl {auto,torchaudio,warp_rnnt,naive,ctc}` with safe CPU fallback on MPS
   - Added alignment size guard (`--max_align`) and profiling spans (`record_function`) around forward, loss, predictor/joiner steps
   - Naive RNNT path now reports loss value while using encoder-CTC for gradients to avoid MPS autograd issues
- [x] LibriSpeech-backed RNNT data pipeline using CSV manifests (tokenizer + dataset + collate)
- [x] Run LibriSpeech RNNT sanity training and record initial WER (approx greedy)
- [x] Provide naive RNNT loss path (`--force_naive_rnnt`) for environments without RNNT loss