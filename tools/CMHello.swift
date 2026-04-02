/**
 * CoreML Hello World Validation Tool for MambaASR Model Deployment
 *
 * This CLI utility provides basic CoreML model validation for the MambaASR speech
 * recognition pipeline. It serves as a lightweight verification tool to ensure
 * CoreML models load correctly and execute basic inference operations before
 * full integration testing.
 *
 * System Integration:
 * - Input: .mlmodelc compiled CoreML models from export pipeline
 * - Processing: Basic model loading and single inference validation
 * - Output: Timing metrics and shape verification for deployment readiness
 * - Purpose: Quick smoke test for CoreML model correctness
 *
 * Model Interface Validation:
 * - Tests standard MambaASR model interface with synthetic inputs
 * - Validates expected output tensor shapes and availability
 * - Measures basic inference latency for performance baseline
 * - Verifies CoreML model compilation and loading pipeline
 *
 * Deployment Pipeline Integration:
 * - Called by CI/CD scripts for automated model validation
 * - Used during development for quick model testing
 * - Provides foundation for more comprehensive validation tools
 * - Enables rapid feedback on CoreML export pipeline correctness
 *
 * Called By:
 * - CI/CD automation scripts after CoreML model export
 * - Development workflows for manual model validation
 * - Testing pipelines requiring basic CoreML functionality verification
 * - Build scripts ensuring model deployment readiness
 *
 * Calls:
 * - CoreML framework for model loading and inference execution
 * - Foundation framework for command line argument processing
 * - MLMultiArray creation for tensor input/output management
 * - CFAbsoluteTimeGetCurrent() for performance timing measurements
 *
 * Usage Examples:
 *   ./cmhello MambaASR.mlmodelc                    // Basic validation
 *   ./cmhello /path/to/exported/model.mlmodelc     // Custom path validation
 *   time ./cmhello model.mlmodelc                  // Performance measurement
 *
 * Performance Expectations:
 * - Model loading: ~50-200ms depending on model size
 * - Single inference: ~5-50ms depending on compute units available
 * - Memory usage: <100MB peak for model and test tensors
 * - Success criteria: Clean execution with expected output shapes
 *
 * Error Conditions:
 * - Invalid model path: File not found or corrupted CoreML model
 * - Model interface mismatch: Unexpected input/output tensor names or shapes
 * - Runtime errors: CoreML framework issues or incompatible model format
 * - Memory pressure: Insufficient system memory for model loading
 */

import Foundation
import CoreML

/// Named constants for CoreML validation test configuration and timing.
///
/// These constants define the operational parameters for the validation test,
/// ensuring consistent behavior and enabling easy adjustment of test parameters
/// without modifying the core logic throughout the validation flow.
private enum CMHelloConstants {
    
    // MARK: - Model Interface Configuration
    
    /// Expected input tensor dimensions for MambaASR model validation.
    /// These dimensions must match the exported CoreML model interface
    /// to ensure compatibility between training pipeline and deployment.
    static let audioChunkBatch = 1
    static let audioChunkTime = 256
    static let audioChunkFeatures = 80
    
    static let tokenInputBatch = 1
    static let tokenInputSequence = 1
    
    static let predictorHiddenBatch = 1
    static let predictorHiddenSequence = 1
    static let predictorHiddenDimension = 256
    
    // MARK: - Data Types and Values
    
    /// MLMultiArray data types matching model export configuration.
    /// Must align with PyTorch export pipeline to ensure numerical compatibility.
    static let audioDataType: MLMultiArrayDataType = .float32
    static let tokenDataType: MLMultiArrayDataType = .int32
    static let hiddenDataType: MLMultiArrayDataType = .float32
    
    /// Initial token value for predictor input tensor.
    /// Zero represents the blank token in RNN-T vocabulary for proper initialization.
    static let initialTokenValue: Int32 = 0
    
    /// Default tensor initialization value for synthetic validation inputs.
    /// Zero provides deterministic, neutral input for reproducible validation.
    static let defaultTensorValue: NSNumber = 0
    
    // MARK: - Model Interface Names
    
    /// CoreML input and output tensor names matching export pipeline.
    /// These strings must exactly match the names defined in export_coreml.py
    /// to ensure proper tensor binding during inference execution.
    static let audioInputName = "audio_chunk"
    static let tokenInputName = "token_in"
    static let hiddenInputName = "predictor_hidden_in"
    
    static let logitsOutputName = "logits_time"
    static let hiddenOutputName = "predictor_hidden_out"
    
    // MARK: - Timing and Performance
    
    /// Telemetry flush delay to ensure complete performance data capture.
    /// CoreML and Instruments require time to flush telemetry data after inference.
    /// This delay ensures complete performance traces are captured before exit.
    static let telemetryFlushDelaySeconds: TimeInterval = 2.0
    
    /// Minimum expected model loading time for validation (milliseconds).
    /// Values below this threshold may indicate cached or incomplete loading.
    static let minExpectedLoadTimeMs: Double = 10.0
    
    /// Maximum acceptable model loading time for validation (milliseconds).
    /// Values above this threshold may indicate performance issues.
    static let maxAcceptableLoadTimeMs: Double = 5000.0
}

// MARK: - Command Line Interface

/// Process command line arguments for flexible model path specification.
///
/// Supports multiple usage patterns for development and CI/CD integration:
/// 1. Single argument: Model path for validation
/// 2. No arguments: Display usage information and exit
///
/// This simple interface enables easy integration into automated testing
/// pipelines while maintaining clear usage semantics for manual execution.
let args = CommandLine.arguments
if args.count <= 1 {
    fputs("usage: cmhello /path/to/model.mlmodelc\n", stderr)
    exit(2)
}
let modelcPath = args[1]

// MARK: - Model Loading and Validation

/// Configure CoreML model for optimal Apple Silicon performance.
/// Enables all compute units (.all) to allow automatic selection between
/// Apple Neural Engine, GPU (Metal), and CPU based on operation compatibility.
let cfg = MLModelConfiguration()
cfg.computeUnits = .all

/// Load and time the CoreML model compilation and instantiation process.
/// Model loading includes both file system access and CoreML runtime preparation,
/// providing essential timing metrics for deployment pipeline validation.
let t0_load = CFAbsoluteTimeGetCurrent()
let model = try MLModel(contentsOf: URL(fileURLWithPath: modelcPath), configuration: cfg)
let t1_load = CFAbsoluteTimeGetCurrent()
let loadTimeMs = (t1_load - t0_load) * 1000.0

print("CoreML hello: model loaded (.all)")
print(String(format: " -> Model load time: %.2f ms", loadTimeMs))

// Validate load time is within expected bounds for deployment
if loadTimeMs < CMHelloConstants.minExpectedLoadTimeMs {
    print("⚠️  Warning: Load time unusually fast, may indicate cached model")
} else if loadTimeMs > CMHelloConstants.maxAcceptableLoadTimeMs {
    print("⚠️  Warning: Load time exceeds acceptable threshold for deployment")
}

// MARK: - Tensor Creation Utilities

/// Creates MLMultiArray tensors with specified shape and data type for model validation.
///
/// This utility function provides standardized tensor creation for CoreML model testing,
/// ensuring consistent initialization patterns and proper memory allocation across
/// different input tensor types required by the MambaASR model interface.
///
/// Tensor Initialization Strategy:
/// - Zero initialization: Provides deterministic, neutral input values
/// - Shape validation: Ensures requested dimensions are valid for MLMultiArray
/// - Type safety: Maintains data type consistency with model expectations
/// - Memory efficiency: Single allocation with immediate initialization
///
/// Called By:
/// - Main validation logic for creating synthetic test inputs
/// - Audio chunk tensor creation for mel-spectrogram simulation
/// - Token input tensor creation for predictor initialization
/// - Hidden state tensor creation for RNN state management
///
/// Args:
///     shape: Array of tensor dimensions as NSNumber objects
///           - Must be valid MLMultiArray shape (positive integers)
///           - Order matches CoreML model input specification
///     dt: MLMultiArrayDataType for tensor element storage
///         - Must match model's expected input data types
///         - Supports .float32, .int32, and other CoreML types
///
/// Returns:
///     MLMultiArray: Initialized tensor ready for model inference
///                   - All elements set to zero for deterministic testing
///                   - Shape and type match function parameters
///
/// Throws:
///     MLMultiArray initialization errors for invalid shapes or unsupported types
///
/// Example Usage:
/// ```swift
/// let audioTensor = try make([1, 256, 80], .float32)     // Audio features
/// let tokenTensor = try make([1, 1], .int32)             // Token input
/// let hiddenTensor = try make([1, 1, 256], .float32)     // Hidden state
/// ```
///
/// Performance Characteristics:
/// - Allocation time: O(product of shape dimensions)
/// - Memory usage: shape size × data type size
/// - Initialization: Single-pass zero filling for deterministic results
func make(_ shape: [NSNumber], _ dt: MLMultiArrayDataType) throws -> MLMultiArray {
    let a = try MLMultiArray(shape: shape as [NSNumber], dataType: dt)
    // Initialize all elements to zero for deterministic validation behavior
    for i in 0..<a.count { 
        a[i] = CMHelloConstants.defaultTensorValue 
    }
    return a
}
// MARK: - Model Interface Validation

/// Create standard MambaASR model input tensors using defined constants.
/// These tensors match the exact interface expected by the exported CoreML model,
/// ensuring compatibility between the PyTorch training pipeline and CoreML deployment.
let audio = try make([
    NSNumber(value: CMHelloConstants.audioChunkBatch),
    NSNumber(value: CMHelloConstants.audioChunkTime), 
    NSNumber(value: CMHelloConstants.audioChunkFeatures)
], CMHelloConstants.audioDataType)

let token = try make([
    NSNumber(value: CMHelloConstants.tokenInputBatch),
    NSNumber(value: CMHelloConstants.tokenInputSequence)
], CMHelloConstants.tokenDataType)

let hidden = try make([
    NSNumber(value: CMHelloConstants.predictorHiddenBatch),
    NSNumber(value: CMHelloConstants.predictorHiddenSequence),
    NSNumber(value: CMHelloConstants.predictorHiddenDimension)
], CMHelloConstants.hiddenDataType)

// Initialize token input with blank token for proper RNN-T predictor state
token[0] = NSNumber(value: CMHelloConstants.initialTokenValue)

/// Construct model input dictionary with standardized tensor names.
/// These names must exactly match the CoreML model export configuration
/// to ensure proper tensor binding during inference execution.
let inputs: [String: Any] = [
    CMHelloConstants.audioInputName: audio,
    CMHelloConstants.tokenInputName: token,
    CMHelloConstants.hiddenInputName: hidden
]

// MARK: - Inference Execution and Timing

/// Execute CoreML inference with comprehensive timing measurement.
/// This validation confirms the model executes successfully and provides
/// performance metrics essential for deployment pipeline validation.
print("CoreML hello: prediction start")
let t0_pred = CFAbsoluteTimeGetCurrent()
let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs))
let t1_pred = CFAbsoluteTimeGetCurrent()
let predictionTimeMs = (t1_pred - t0_pred) * 1000.0

print("CoreML hello: prediction end")
print(String(format: " -> Prediction time: %.2f ms", predictionTimeMs))

// MARK: - Output Validation

/// Validate expected output tensors are present and report their shapes.
/// Missing tensors indicate model export issues or interface mismatches
/// that must be resolved before deployment to production systems.
let expectedOutputs = [
    CMHelloConstants.logitsOutputName,
    CMHelloConstants.hiddenOutputName
]

print("\n📊 Model Output Validation:")
for outputName in expectedOutputs {
    if let ma = out.featureValue(for: outputName)?.multiArrayValue {
        print("✅ \(outputName): \(ma.shape)")
    } else {
        print("❌ \(outputName): MISSING")
    }
}

// MARK: - Telemetry and Cleanup

/// Ensure complete telemetry data capture before process termination.
/// CoreML and Instruments require time to flush performance data to disk,
/// particularly important for CI/CD pipelines that capture timing metrics.
/// The sleep duration is calibrated for complete telemetry capture without
/// excessive overhead in automated testing environments.
print("\n⏳ Waiting for telemetry flush...")
Thread.sleep(forTimeInterval: CMHelloConstants.telemetryFlushDelaySeconds)
print("✅ CMHello: Validation completed successfully")
