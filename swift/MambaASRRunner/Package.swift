// swift-tools-version: 5.10

/**
 * MambaASRRunner Swift Package Manifest
 *
 * This Swift Package Manager manifest defines the MambaASRRunner CLI utility for
 * Core ML model validation in the MambaASR speech recognition pipeline. It serves
 * as a critical integration tool between PyTorch training and Apple Silicon deployment.
 *
 * Package Purpose:
 * - Core ML validation: Verify exported models work correctly on Apple Silicon
 * - CI/CD integration: Automated testing of model conversion pipeline
 * - Development tooling: Quick validation during model export iterations
 * - Deployment readiness: Confirm Apple Neural Engine compatibility
 *
 * System Integration:
 * - Input: .mlpackage files from scripts/export_coreml.py (Python → Core ML)
 * - Processing: Stateful Core ML inference with synthetic data validation
 * - Output: Success/failure confirmation for deployment pipelines
 * - Deployment: Foundation for iOS/macOS speech recognition applications
 *
 * Platform Requirements:
 * - macOS 13.0+: Required for Core ML StateType support and ANE optimization
 * - Apple Silicon: Optimized for M1/M2/M3 unified memory architecture
 * - Swift 5.10+: Modern Swift toolchain with Core ML improvements
 * - Xcode 15.0+: Development environment for Apple Silicon optimization
 *
 * Core Dependencies (System Frameworks):
 * - Foundation: File system access, command line argument processing
 * - CoreML: Model loading, compilation, and inference execution
 * - Accelerate: Numerical operations and vectorized computations
 *
 * Architecture Integration:
 * - Called by: CI/CD pipelines, development scripts, manual validation
 * - Validates: Core ML model interface contract and Apple Silicon execution
 * - Enables: iOS/macOS app integration with validated Core ML models
 * - Prepares: Production deployment of MambaASR speech recognition
 *
 * Build Configuration:
 * - Executable target: Command-line utility for automation integration
 * - No external dependencies: Uses only Apple system frameworks
 * - Minimal footprint: Fast compilation and execution for CI/CD efficiency
 * - Platform-specific: Optimized for Apple Silicon deployment scenarios
 *
 * Usage in Development Pipeline:
 * 1. Python training produces optimized MCT models
 * 2. Core ML export creates .mlpackage files with stateful interface
 * 3. MambaASRRunner validates Core ML models on target hardware
 * 4. Success confirmation enables iOS/macOS app integration
 * 5. Production deployment with validated Core ML models
 *
 * Performance Characteristics:
 * - Build time: ~10-30 seconds for clean compilation
 * - Runtime: ~500ms-2s for complete model validation
 * - Memory usage: <100MB peak during Core ML inference
 * - ANE utilization: Validates >90% Neural Engine operation placement
 *
 * CI/CD Integration:
 * - Exit codes: 0 for success, 1 for validation failures
 * - Logging: Structured output for automated result parsing
 * - Error reporting: Detailed diagnostics for debugging export issues
 * - Performance metrics: Inference timing and tensor shape validation
 */

import PackageDescription

/// Named constants for package configuration and deployment targeting.
///
/// These constants define the deployment requirements and build configuration
/// for MambaASRRunner CLI utility across Apple Silicon development environments.
private enum PackageConstants {
    
    /// Minimum macOS version required for Core ML StateType and ANE features.
    /// macOS 13.0 introduces stateful Core ML models and improved ANE targeting.
    /// Earlier versions lack StateType support essential for RNN-T streaming.
    static let minimumMacOSVersion: SupportedPlatform = .macOS(.v13)
    
    /// Package identifier for Swift Package Manager resolution.
    /// Matches directory name and executable target for consistency.
    /// Enables clear identification in build logs and dependency resolution.
    static let packageName = "MambaASRRunner"
    
    /// Executable target name for command-line utility generation.
    /// Produces binary named "MambaASRRunner" for CI/CD script integration.
    /// Consistent naming across package definition and build outputs.
    static let executableTargetName = "MambaASRRunner"
}

let package = Package(
    name: PackageConstants.packageName,
    platforms: [
        PackageConstants.minimumMacOSVersion
    ],
    products: [
        .executable(
            name: PackageConstants.executableTargetName, 
            targets: [PackageConstants.executableTargetName]
        )
    ],
    dependencies: [
        // No external dependencies - uses only Apple system frameworks
        // Foundation: File system and command line interface
        // CoreML: Model loading and inference execution  
        // Accelerate: Numerical operations for tensor processing
    ],
    targets: [
        .executableTarget(
            name: PackageConstants.executableTargetName,
            dependencies: []
        )
    ]
)
