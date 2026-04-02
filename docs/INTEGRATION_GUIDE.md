# MambaASR Integration Guide for AI-First Development

## Overview

This guide provides comprehensive integration patterns and examples for AI developers working with the MambaASR system. Every integration point is documented with concrete examples, error handling patterns, and performance considerations.

## Core Integration Patterns

### Pattern 1: Training Pipeline Integration

#### Basic Training Workflow

```python
# File: custom_training_script.py
# Integration with: train_RNNT.py, modules/mct/mct_model.py, config/

from config import MambaASRConfig
from modules.mct.mct_model import MCTModel, MCTConfig
from modules.rnnt_loss_mps import rnnt_loss_mps
from datasets.librispeech_csv import LibriSpeechCSVDataset

def integrated_training_example():
    \"\"\"Example of complete training pipeline integration.
    
    This demonstrates the canonical integration pattern for training
    MambaASR models with full Apple Silicon optimization.
    \"\"\"
    
    # Step 1: Configuration setup with environment override support
    config = MCTConfig(
        d_model=MambaASRConfig.Model.DEFAULT_D_MODEL,
        n_blocks=MambaASRConfig.Model.DEFAULT_N_BLOCKS,
        joint_dim=MambaASRConfig.Model.DEFAULT_JOINT_DIM,
        vocab_size=MambaASRConfig.Tokenizer.VOCABULARY_SIZE
    )
    
    # Step 2: Model instantiation with device optimization
    device = AppleSiliconConfig.Performance.get_optimal_device()
    model = MCTModel(config).to(device)
    
    # Step 3: Dataset integration
    dataset = LibriSpeechCSVDataset(
        manifest_path="train.csv",
        sample_rate=MambaASRConfig.Model.SAMPLE_RATE,
        max_duration=10.0
    )
    
    # Step 4: Loss function integration with backend selection
    def compute_loss(logits, targets, input_lengths, target_lengths):
        loss, gradients, backend = rnnt_loss_mps(
            logits, targets, input_lengths, target_lengths,
            max_align=MambaASRConfig.RNNTLoss.DEFAULT_MAX_ALIGNMENT
        )
        
        # Handle explicit gradients if CPU fallback used
        if gradients is not None:
            logits.backward(gradients)
        else:
            loss.backward()
        
        return loss, backend
    
    # Step 5: Training loop with Apple Silicon optimizations
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=MambaASRConfig.Training.DEFAULT_LEARNING_RATE
    )
    
    for batch in DataLoader(dataset, batch_size=MambaASRConfig.Training.DEFAULT_BATCH_SIZE):
        # Forward pass
        logits = model(batch['features'], batch['feature_lengths'])
        
        # Loss computation with automatic backend selection
        loss, backend = compute_loss(
            logits, batch['tokens'], 
            batch['feature_lengths'], batch['token_lengths']
        )
        
        # Optimization step
        optimizer.step()
        optimizer.zero_grad(set_to_none=MambaASRConfig.Training.GRAD_SET_TO_NONE)
        
        print(f"Loss: {loss.item():.4f}, Backend: {backend}")
```

### Pattern 2: Model Export and Deployment Integration

#### Core ML Export Workflow

```python
# File: custom_export_script.py
# Integration with: scripts/export_coreml.py, swift/MambaASRRunner/

import coremltools as ct
from modules.mct.mct_model import MCTModel
from config import AppleSiliconConfig

def integrated_export_example():
    \"\"\"Example of complete export and validation workflow.
    
    This demonstrates the canonical pattern for PyTorch → Core ML → Swift
    deployment with comprehensive validation.
    \"\"\"
    
    # Step 1: Load trained PyTorch model
    checkpoint = torch.load("checkpoints/trained_model.pt", map_location="cpu", weights_only=True)
    model = MCTModel.from_checkpoint(checkpoint)
    model.eval()
    
    # Step 2: Prepare for Core ML conversion
    example_input = torch.randn(1, 1000, 80)  # (batch, time, features)
    traced_model = torch.jit.trace(model, example_input)
    
    # Step 3: Core ML conversion with ANE optimization
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=(1, ct.RangeDim(1, 10000), 80))],
        compute_units=ct.ComputeUnit.ALL,  # Enable ANE
        minimum_deployment_target=ct.target.iOS15  # ANE support
    )
    
    # Step 4: Save with metadata for Swift integration
    coreml_model.save("exports/MambaASR.mlpackage")
    
    # Step 5: Validation via Swift runner integration
    validation_result = subprocess.run([
        "swift/MambaASRRunner/.build/release/MambaASRRunner",
        "--mlpackage", "exports/MambaASR.mlpackage",
        "--wav", "test_audio.wav",
        "--stream"
    ], capture_output=True, text=True)
    
    if validation_result.returncode == 0:
        print("✅ Core ML export and Swift validation successful")
        print(f"Transcript: {validation_result.stdout}")
    else:
        print("❌ Validation failed:", validation_result.stderr)
```

### Pattern 3: Evaluation and Metrics Integration

#### Comprehensive Evaluation Pipeline

```python
# File: custom_evaluation_script.py
# Integration with: scripts/compute_wer_cer.py, utils/metrics.py

from utils.metrics import wer, batch_wer
from scripts.compute_wer_cer import normalize_text_for_eval, extract_text
import subprocess
from pathlib import Path

def integrated_evaluation_example():
    \"\"\"Example of complete evaluation pipeline integration.
    
    This demonstrates the canonical pattern for model evaluation
    across multiple test sets with automated reporting.
    \"\"\"
    
    # Step 1: Generate transcripts via Swift runner
    test_files = list(Path("test_audio/").glob("*.wav"))
    transcript_results = []
    
    for audio_file in test_files:
        # Run Swift inference
        result = subprocess.run([
            "swift/MambaASRRunner/.build/release/MambaASRRunner",
            "--mlpackage", "exports/MambaASR.mlpackage",
            "--wav", str(audio_file),
            "--stream"
        ], capture_output=True, text=True)
        
        # Extract transcript using standardized parser
        transcript = extract_text_from_runner_output(result.stdout)
        transcript_results.append({
            'file': audio_file.stem,
            'transcript': transcript,
            'reference': load_reference_transcript(audio_file.stem)
        })
    
    # Step 2: Compute evaluation metrics
    references = [r['reference'] for r in transcript_results]
    hypotheses = [r['transcript'] for r in transcript_results]
    
    # Use centralized metrics computation
    overall_wer = batch_wer(references, hypotheses)
    
    # Step 3: Per-file analysis
    detailed_results = []
    for result in transcript_results:
        file_wer = wer(result['reference'], result['transcript'])
        detailed_results.append({
            'file': result['file'],
            'wer': file_wer,
            'reference': result['reference'],
            'hypothesis': result['transcript'],
            'interpretation': MambaASRConfig.Metrics.get_wer_interpretation(file_wer)
        })
    
    # Step 4: Generate comprehensive report
    generate_evaluation_report(overall_wer, detailed_results)

def extract_text_from_runner_output(stdout: str) -> str:
    \"\"\"Extract transcript from MambaASRRunner stdout.
    
    Integration point with scripts/compute_wer_cer.py parsing logic.
    \"\"\"
    lines = stdout.strip().split('\\n')
    for line in lines:
        if 'transcript:' in line.lower() and '(ids)' not in line.lower():
            try:
                return line.split(':', 1)[1].strip()
            except IndexError:
                continue
    return ""

def generate_evaluation_report(overall_wer: float, detailed_results: list):
    \"\"\"Generate markdown evaluation report.
    
    Integration with reporting patterns used throughout the system.
    \"\"\"
    report_lines = [
        "# MambaASR Evaluation Report",
        f"**Overall WER:** {overall_wer:.4f} {MambaASRConfig.Metrics.get_wer_interpretation(overall_wer)}",
        "",
        "## Detailed Results",
        "| File | WER | Interpretation | Reference | Hypothesis |",
        "|------|-----|----------------|-----------|------------|"
    ]
    
    for result in detailed_results:
        report_lines.append(
            f"| {result['file']} | {result['wer']:.4f} | {result['interpretation']} | "
            f"{result['reference'][:50]}... | {result['hypothesis'][:50]}... |"
        )
    
    with open("evaluation_report.md", "w") as f:
        f.write("\\n".join(report_lines))
```

## Advanced Integration Patterns

### Pattern 4: Configuration-Driven Component Integration

```python
# File: configurable_pipeline.py
# Integration with: config/ module, environment variable overrides

from config import MambaASRConfig, AppleSiliconConfig, EnvironmentConfig

class ConfigurablePipeline:
    \"\"\"Example of configuration-driven component integration.
    
    This demonstrates how to build flexible pipelines that adapt
    to different deployment scenarios via configuration.
    \"\"\"
    
    def __init__(self):
        # Step 1: Initialize configuration with environment overrides
        self.setup_environment()
        self.training_config = MambaASRConfig.Training
        self.model_config = MambaASRConfig.Model
        self.apple_silicon_config = AppleSiliconConfig
        
        # Step 2: Log configuration for debugging
        print(EnvironmentConfig.get_environment_summary())
        print(AppleSiliconConfig.get_apple_silicon_summary())
    
    def setup_environment(self):
        \"\"\"Configure environment based on deployment scenario.\"\"\"
        deployment_mode = os.getenv("DEPLOYMENT_MODE", "development")
        
        if deployment_mode == "development":
            EnvironmentConfig.set_development_defaults()
        elif deployment_mode == "production":
            EnvironmentConfig.set_production_defaults()
        
        # Apply Apple Silicon optimizations
        AppleSiliconConfig.setup_apple_silicon_environment()
    
    def create_model(self):
        \"\"\"Create model with configuration-driven parameters.\"\"\"
        config = MCTConfig(
            d_model=EnvironmentConfig.get_environment_value("MAMBA_D_MODEL"),
            n_blocks=EnvironmentConfig.get_environment_value("MAMBA_N_BLOCKS"),
            joint_dim=EnvironmentConfig.get_environment_value("MAMBA_JOINT_DIM"),
            vocab_size=EnvironmentConfig.get_environment_value("MAMBA_VOCAB_SIZE")
        )
        
        device = self.apple_silicon_config.Performance.get_optimal_device()
        return MCTModel(config).to(device)
    
    def create_training_setup(self):
        \"\"\"Create training components with configuration integration.\"\"\"
        batch_size = EnvironmentConfig.get_environment_value("MAMBA_BATCH_SIZE")
        learning_rate = EnvironmentConfig.get_environment_value("MAMBA_LEARNING_RATE")
        
        dataset = LibriSpeechCSVDataset(
            manifest_path=os.getenv("TRAINING_MANIFEST", "train.csv"),
            max_duration=10.0
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            num_workers=0,  # Apple Silicon optimization
            pin_memory=False  # Unified memory architecture
        )
        
        model = self.create_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        return model, optimizer, dataloader
```

### Pattern 5: Error-Resilient Integration

```python
# File: resilient_integration.py
# Integration with: error handling patterns across all components

class ResilientPipeline:
    \"\"\"Example of error-resilient integration patterns.
    
    This demonstrates comprehensive error handling and recovery
    strategies for production deployment.
    \"\"\"
    
    def __init__(self):
        self.device = self.setup_device_with_fallback()
        self.rnnt_backend = self.setup_rnnt_with_fallback()
        self.metrics_tracker = MetricsTracker()
    
    def setup_device_with_fallback(self):
        \"\"\"Device setup with comprehensive fallback chain.\"\"\"
        try:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
                # Verify MPS functionality
                test_tensor = torch.randn(10, 10, device=device)
                test_result = torch.matmul(test_tensor, test_tensor.T)
                self.metrics_tracker.log("device_selected", "mps")
                return device
        except Exception as e:
            self.metrics_tracker.log("mps_fallback", str(e))
        
        try:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.metrics_tracker.log("device_selected", "cuda")
                return device
        except Exception as e:
            self.metrics_tracker.log("cuda_fallback", str(e))
        
        self.metrics_tracker.log("device_selected", "cpu")
        return torch.device("cpu")
    
    def setup_rnnt_with_fallback(self):
        \"\"\"RNN-T backend setup with fallback chain.\"\"\"
        from modules.rnnt_loss_mps import select_best_backend
        
        backend_fn, backend_name = select_best_backend()
        self.metrics_tracker.log("rnnt_backend_selected", backend_name)
        
        if backend_fn is None:
            self.metrics_tracker.log("rnnt_fallback", "no_backend_available")
            # Implement emergency CTC fallback
            return self.setup_ctc_fallback()
        
        return backend_fn, backend_name
    
    def train_with_error_recovery(self, model, dataloader, optimizer):
        \"\"\"Training loop with comprehensive error recovery.\"\"\"
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Attempt training step
                loss = self.training_step(model, batch, optimizer)
                self.metrics_tracker.log("training_step_success", batch_idx)
                consecutive_failures = 0  # Reset failure counter
                
            except RuntimeError as e:
                consecutive_failures += 1
                self.metrics_tracker.log("training_step_failure", {
                    "batch_idx": batch_idx,
                    "error": str(e),
                    "consecutive_failures": consecutive_failures
                })
                
                # Handle specific error types
                if "out of memory" in str(e).lower():
                    self.handle_memory_error(batch_idx)
                elif "mps" in str(e).lower():
                    self.handle_mps_error(batch_idx)
                
                # Circuit breaker pattern
                if consecutive_failures >= max_consecutive_failures:
                    self.metrics_tracker.log("training_circuit_breaker", batch_idx)
                    raise RuntimeError(f"Too many consecutive failures at batch {batch_idx}")
                
                # Skip this batch and continue
                continue
    
    def handle_memory_error(self, batch_idx):
        \"\"\"Handle memory pressure with graduated response.\"\"\"
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Reduce batch size for subsequent batches if needed
        self.metrics_tracker.log("memory_pressure_handled", batch_idx)
    
    def handle_mps_error(self, batch_idx):
        \"\"\"Handle MPS-specific errors with fallback.\"\"\"
        # Force fallback to CPU for this operation
        self.metrics_tracker.log("mps_error_fallback", batch_idx)
        # Implementation would include moving tensors to CPU temporarily

class MetricsTracker:
    \"\"\"Centralized metrics tracking for error analysis.\"\"\"
    
    def __init__(self):
        self.metrics = {}
        self.log_file = "pipeline_metrics.json"
    
    def log(self, metric_name: str, value):
        \"\"\"Log metric with timestamp for analysis.\"\"\"
        import time
        
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append({
            "timestamp": time.time(),
            "value": value
        })
        
        # Periodic save
        if len(self.metrics) % 100 == 0:
            self.save_metrics()
    
    def save_metrics(self):
        \"\"\"Save metrics to file for analysis.\"\"\"
        import json
        with open(self.log_file, "w") as f:
            json.dump(self.metrics, f, indent=2)
```

## Component-Specific Integration Examples

### Swift Runtime Integration

```swift
// File: MambaASRIntegration.swift
// Integration with: swift/MambaASRRunner/, Core ML models

import Foundation
import CoreML

class MambaASRIntegration {
    /// Example of Swift integration with MambaASR Core ML models.
    /// This demonstrates the canonical pattern for iOS/macOS app integration.
    
    private let model: MLModel
    private let configuration: MLModelConfiguration
    
    init() throws {
        // Step 1: Configure Core ML for optimal Apple Silicon performance
        configuration = MLModelConfiguration()
        configuration.computeUnits = .all  // Enable ANE + GPU + CPU
        configuration.allowLowPrecisionAccumulationOnGPU = true
        
        // Step 2: Load compiled Core ML model
        let modelURL = Bundle.main.url(forResource: "MambaASR", withExtension: "mlmodelc")!
        model = try MLModel(contentsOf: modelURL, configuration: configuration)
    }
    
    func processAudio(_ audioData: Data) throws -> String {
        // Step 3: Preprocessing integration
        let melFeatures = try preprocessAudio(audioData)
        
        // Step 4: Model inference with error handling
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "audio_features": MLMultiArray(melFeatures)
        ])
        
        let output = try model.prediction(from: input)
        
        // Step 5: Postprocessing integration
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw MambaASRError.invalidOutput
        }
        
        return try decodeLogitsToText(logits)
    }
    
    private func preprocessAudio(_ audioData: Data) throws -> [[Float]] {
        // Integration with mel-spectrogram computation
        // This would call the same preprocessing used in training
        // Implementation details follow Core ML input requirements
        return []
    }
    
    private func decodeLogitsToText(_ logits: MLMultiArray) throws -> String {
        // Integration with character tokenizer vocabulary
        // This uses the same 29-character vocabulary from training
        return ""
    }
}

enum MambaASRError: Error {
    case invalidOutput
    case preprocessingFailed
    case decodingFailed
}
```

## Integration Testing Patterns

### Comprehensive Integration Test Suite

```python
# File: test_integration.py
# Integration with: pytest framework, all system components

import pytest
from pathlib import Path
import subprocess
import json

class TestSystemIntegration:
    \"\"\"Comprehensive integration tests for MambaASR system.
    
    These tests validate end-to-end integration across all components
    and deployment scenarios.
    \"\"\"
    
    @pytest.fixture
    def trained_model_checkpoint(self):
        \"\"\"Provide trained model checkpoint for integration tests.\"\"\"
        checkpoint_path = Path("checkpoints/test_model.pt")
        if not checkpoint_path.exists():
            pytest.skip("Test model checkpoint not available")
        return checkpoint_path
    
    @pytest.fixture
    def test_audio_files(self):
        \"\"\"Provide test audio files for integration validation.\"\"\"
        audio_dir = Path("test_data/audio/")
        audio_files = list(audio_dir.glob("*.wav"))
        if not audio_files:
            pytest.skip("Test audio files not available")
        return audio_files
    
    def test_training_to_export_integration(self, trained_model_checkpoint):
        \"\"\"Test complete training → export → validation pipeline.\"\"\"
        
        # Step 1: Verify checkpoint can be loaded
        checkpoint = torch.load(trained_model_checkpoint, map_location="cpu", weights_only=True)
        assert "model_state_dict" in checkpoint
        
        # Step 2: Export to Core ML
        export_result = subprocess.run([
            "python", "scripts/export_coreml.py",
            "--checkpoint", str(trained_model_checkpoint),
            "--output", "test_exports/integration_test.mlpackage"
        ], capture_output=True, text=True)
        
        assert export_result.returncode == 0, f"Export failed: {export_result.stderr}"
        
        # Step 3: Validate with Swift runner
        validation_result = subprocess.run([
            "swift/MambaASRRunner/.build/release/MambaASRRunner",
            "--mlpackage", "test_exports/integration_test.mlpackage",
            "--wav", "test_data/audio/sample.wav",
            "--stream"
        ], capture_output=True, text=True)
        
        assert validation_result.returncode == 0, f"Validation failed: {validation_result.stderr}"
        assert "transcript:" in validation_result.stdout.lower()
    
    def test_configuration_integration(self):
        \"\"\"Test configuration system integration across components.\"\"\"
        
        # Step 1: Set environment overrides
        os.environ["MAMBA_BATCH_SIZE"] = "4"
        os.environ["MAMBA_D_MODEL"] = "512"
        
        # Step 2: Verify configuration propagation
        overrides = EnvironmentConfig.get_all_environment_overrides()
        assert "MAMBA_BATCH_SIZE" in overrides
        assert overrides["MAMBA_BATCH_SIZE"] == 4
        
        # Step 3: Verify component initialization uses overrides
        training_config = MambaASRConfig.Training()
        model_config = MambaASRConfig.Model()
        
        # Configuration should reflect environment overrides
        batch_size = EnvironmentConfig.get_environment_value("MAMBA_BATCH_SIZE")
        d_model = EnvironmentConfig.get_environment_value("MAMBA_D_MODEL")
        
        assert batch_size == 4
        assert d_model == 512
    
    def test_evaluation_pipeline_integration(self, test_audio_files):
        \"\"\"Test complete evaluation pipeline integration.\"\"\"
        
        # Step 1: Generate transcripts
        for audio_file in test_audio_files[:3]:  # Test subset
            result = subprocess.run([
                "swift/MambaASRRunner/.build/release/MambaASRRunner",
                "--mlpackage", "exports/MambaASR.mlpackage",
                "--wav", str(audio_file),
                "--stream"
            ], capture_output=True, text=True)
            
            # Save transcript for evaluation
            transcript_file = f"test_transcripts/{audio_file.stem}.txt"
            with open(transcript_file, "w") as f:
                f.write(result.stdout)
        
        # Step 2: Compute evaluation metrics
        eval_result = subprocess.run([
            "python", "scripts/compute_wer_cer.py",
            "--ref", "test_data/references.txt",
            "--glob", "test_transcripts/*.txt",
            "--out", "test_evaluation_report.md"
        ], capture_output=True, text=True)
        
        assert eval_result.returncode == 0
        assert Path("test_evaluation_report.md").exists()
    
    def test_apple_silicon_optimization_integration(self):
        \"\"\"Test Apple Silicon optimization integration.\"\"\"
        
        # Step 1: Verify MPS availability detection
        hardware_info = AppleSiliconConfig.Performance.detect_apple_silicon()
        assert "is_apple_silicon" in hardware_info
        
        # Step 2: Verify device selection logic
        optimal_device = AppleSiliconConfig.Performance.get_optimal_device()
        assert str(optimal_device) in ["mps", "cuda", "cpu"]
        
        # Step 3: Verify memory management integration
        memory_info = AppleSiliconConfig.MPS.get_memory_info()
        if memory_info["available"]:
            assert "allocated_mb" in memory_info
            assert "high_watermark_ratio" in memory_info
    
    def test_error_recovery_integration(self):
        \"\"\"Test error recovery and fallback integration.\"\"\"
        
        # Step 1: Test RNN-T backend fallback
        from modules.rnnt_loss_mps import select_best_backend
        backend_fn, backend_name = select_best_backend()
        
        # Should always return a valid backend or None with graceful handling
        assert backend_name in ["torchaudio", "warp_rnnt", "none"]
        
        # Step 2: Test configuration validation
        invalid_batch_size = EnvironmentConfig.get_environment_value(
            "MAMBA_BATCH_SIZE", 
            default=2, 
            validate=True
        )
        
        # Should return valid value even with invalid environment setting
        assert isinstance(invalid_batch_size, int)
        assert invalid_batch_size >= 1
```

This integration guide provides comprehensive patterns for AI developers to understand and extend the MambaASR system, with concrete examples for every major integration point and error handling scenario.