"""
Apple Silicon Hardware Configuration for MambaASR Optimization

This module provides specialized configuration constants for Apple Silicon
hardware optimization. It centralizes all MPS (Metal Performance Shaders),
ANE (Apple Neural Engine), and unified memory architecture parameters
required for optimal MambaASR performance on Apple hardware.

Hardware Architecture Understanding:
- Unified Memory: CPU and GPU share same physical memory pool
- MPS Backend: PyTorch GPU acceleration via Metal Performance Shaders
- Apple Neural Engine: Dedicated ML inference acceleration unit  
- AMX Coprocessors: Matrix multiplication acceleration for large tensors
- Memory Bandwidth: 68-800 GB/s depending on Apple Silicon variant

Apple Silicon Optimization Strategy:
- Memory pressure management for unified memory architecture
- MPS fallback mechanisms for unsupported operations
- ANE utilization via Core ML deployment optimization
- Device detection and capability assessment
- Performance profiling and measurement optimization

Integration Points:
- Used by: All training scripts requiring Apple Silicon optimization
- Coordinates with: MPS backend initialization and device management
- Supports: Core ML export pipeline for ANE deployment
- Enables: Hardware-specific performance tuning and monitoring

Cross-File Dependencies:
- Required by: train_RNNT.py, train_CTC.py for device setup
- Used by: Core ML export scripts for ANE optimization
- Coordinates with: Swift MambaASRRunner for deployment validation
- Integrates with: Performance measurement and profiling workflows
"""

import os
import torch
from typing import Optional, Dict, Any


class AppleSiliconConfig:
    """Apple Silicon hardware configuration and optimization parameters.
    
    This class centralizes all hardware-specific configuration for Apple Silicon
    optimization, including MPS backend settings, memory management parameters,
    and ANE deployment configuration.
    """
    
    class MPS:
        """Metal Performance Shaders (MPS) backend configuration.
        
        These constants control PyTorch MPS backend behavior for GPU acceleration
        on Apple Silicon. They manage memory allocation, fallback behavior, and
        performance optimization for the unified memory architecture.
        """
        
        # Memory Management
        HIGH_WATERMARK_RATIO = 0.8
        """Memory pressure threshold for MPS allocation (0.7-0.8 recommended).
        
        Controls memory allocation limits to prevent system-wide memory pressure.
        Environment override: PYTORCH_MPS_HIGH_WATERMARK_RATIO
        - 0.7: Conservative, prevents system swapping
        - 0.8: Balanced performance and stability  
        - 0.9: Aggressive memory usage for maximum performance
        """
        
        # Fallback Configuration
        ENABLE_FALLBACK_DEFAULT = True
        """Enable CPU fallback for unsupported MPS operations.
        
        Environment override: PYTORCH_ENABLE_MPS_FALLBACK
        - True: Automatic CPU fallback (debugging/development)
        - False: Strict MPS mode, fails on unsupported ops (production)
        """
        
        # Performance Optimization
        SYNCHRONIZATION_POINTS = True
        """Enable explicit synchronization for accurate timing measurements."""
        
        EMPTY_CACHE_THRESHOLD = 0.9
        """Memory usage threshold for triggering cache cleanup."""
        
        # Batch Size Optimization
        MIN_BATCH_SIZE = 1
        """Minimum batch size for MPS optimization."""
        
        MAX_BATCH_SIZE = 8
        """Maximum recommended batch size for Apple Silicon memory limits."""
        
        OPTIMAL_BATCH_SIZE = 4
        """Optimal batch size balancing throughput and memory usage."""
        
        # Tensor Operation Thresholds
        MIN_TENSOR_SIZE_GPU = 1000
        """Minimum tensor size to prefer GPU over CPU (dispatch overhead)."""
        
        MAX_SEQUENCE_LENGTH = 8192
        """Maximum sequence length for stable MPS performance."""
        
        @staticmethod
        def get_memory_info() -> Dict[str, Any]:
            """Get current MPS memory usage information."""
            if not torch.backends.mps.is_available():
                return {"available": False, "reason": "MPS not available"}
            
            try:
                allocated = torch.mps.current_allocated_memory()
                cached = torch.mps.driver_allocated_memory() 
                return {
                    "available": True,
                    "allocated_mb": allocated / (1024 * 1024),
                    "cached_mb": cached / (1024 * 1024),
                    "high_watermark_ratio": os.getenv(
                        "PYTORCH_MPS_HIGH_WATERMARK_RATIO", 
                        str(AppleSiliconConfig.MPS.HIGH_WATERMARK_RATIO)
                    )
                }
            except Exception as e:
                return {"available": True, "error": str(e)}
    
    
    class ANE:
        """Apple Neural Engine (ANE) configuration for Core ML deployment.
        
        These parameters control Core ML model compilation and optimization
        for the Apple Neural Engine, enabling maximum on-device inference
        performance for MambaASR models.
        """
        
        # Model Optimization
        TARGET_ANE_UTILIZATION = 0.9
        """Target ANE utilization percentage for optimal performance."""
        
        MAX_MODEL_SIZE_MB = 1000
        """Maximum model size for ANE deployment (memory constraint)."""
        
        # Inference Configuration
        MAX_CHUNK_SIZE_FRAMES = 512
        """Maximum audio chunk size for ANE processing (memory limit)."""
        
        OPTIMAL_CHUNK_SIZE_FRAMES = 256
        """Optimal chunk size balancing latency and throughput."""
        
        # Quantization Settings
        SUPPORT_INT8_QUANTIZATION = True
        """ANE supports INT8 quantization for memory efficiency."""
        
        SUPPORT_INT4_QUANTIZATION = False
        """ANE INT4 quantization support (limited availability)."""
        
        # Performance Targets
        TARGET_LATENCY_MS = 10
        """Target inference latency for real-time processing."""
        
        MAX_ACCEPTABLE_LATENCY_MS = 50
        """Maximum acceptable latency for deployment."""
        
        # Compilation Settings
        COMPUTE_PRECISION = "FLOAT16"
        """Compute precision for ANE optimization (FLOAT16/FLOAT32)."""
        
        @staticmethod
        def get_ane_info() -> Dict[str, Any]:
            """Get Apple Neural Engine availability and capability information."""
            # Note: ANE availability detection requires Core ML runtime
            return {
                "optimization_target": "Apple Neural Engine",
                "max_model_size_mb": AppleSiliconConfig.ANE.MAX_MODEL_SIZE_MB,
                "target_latency_ms": AppleSiliconConfig.ANE.TARGET_LATENCY_MS,
                "supported_precision": AppleSiliconConfig.ANE.COMPUTE_PRECISION,
                "int8_quantization": AppleSiliconConfig.ANE.SUPPORT_INT8_QUANTIZATION
            }
    
    
    class UnifiedMemory:
        """Unified memory architecture optimization configuration.
        
        Apple Silicon's unified memory architecture requires specific optimization
        strategies that differ from traditional discrete GPU systems. These
        constants manage memory allocation and transfer patterns.
        """
        
        # Memory Allocation Strategy
        PREFER_DEVICE_ALLOCATION = True
        """Prefer device-side allocation to minimize transfers."""
        
        AGGRESSIVE_CACHING = True
        """Enable aggressive caching for unified memory efficiency."""
        
        # Transfer Optimization
        MIN_TRANSFER_SIZE = 1024
        """Minimum tensor size to justify explicit device transfer."""
        
        BATCH_TRANSFER_THRESHOLD = 4
        """Number of tensors to batch for efficient transfer."""
        
        # Pressure Management
        MEMORY_PRESSURE_THRESHOLD = 0.85
        """Memory usage threshold for pressure management."""
        
        SWAP_PREVENTION_ENABLED = True
        """Enable active swap prevention mechanisms."""
        
        # Performance Monitoring
        ENABLE_MEMORY_PROFILING = False
        """Enable detailed memory allocation profiling (debug mode)."""
        
        PROFILE_ALLOCATION_STACKS = False
        """Enable allocation stack trace profiling (debug mode)."""
    
    
    class Performance:
        """Apple Silicon performance optimization and monitoring configuration.
        
        These constants control performance measurement, profiling, and optimization
        strategies specific to Apple Silicon hardware characteristics.
        """
        
        # Measurement Configuration
        WARMUP_ITERATIONS = 3
        """Number of warmup iterations for stable performance measurement."""
        
        MEASUREMENT_ITERATIONS = 10
        """Number of measurement iterations for statistical accuracy."""
        
        # Profiling Settings
        ENABLE_AUTOGRAD_PROFILER = False
        """Enable PyTorch autograd profiler (development mode)."""
        
        ENABLE_METAL_CAPTURE = False
        """Enable Metal frame capture for GPU debugging."""
        
        # Optimization Targets
        TARGET_GPU_UTILIZATION = 0.8
        """Target GPU utilization for optimal performance."""
        
        TARGET_MEMORY_BANDWIDTH = 0.7
        """Target memory bandwidth utilization."""
        
        # Performance Thresholds
        MIN_ACCEPTABLE_FPS = 30
        """Minimum frames per second for real-time processing."""
        
        TARGET_FPS = 100
        """Target frames per second for optimal performance."""
        
        # Hardware Detection
        @staticmethod
        def detect_apple_silicon() -> Dict[str, Any]:
            """Detect Apple Silicon hardware and capabilities."""
            import platform
            
            system_info = {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "is_apple_silicon": platform.machine() == "arm64"
            }
            
            # PyTorch backend availability
            system_info.update({
                "mps_available": torch.backends.mps.is_available(),
                "mps_built": torch.backends.mps.is_built(),
                "cuda_available": torch.cuda.is_available(),
                "torch_version": torch.__version__
            })
            
            return system_info
        
        @staticmethod
        def get_optimal_device() -> torch.device:
            """Get optimal PyTorch device for Apple Silicon."""
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
    
    
    @classmethod
    def setup_apple_silicon_environment(cls) -> Dict[str, Any]:
        """Configure environment for optimal Apple Silicon performance.
        
        Returns:
            Configuration summary with applied settings
        """
        config_applied = {}
        
        # Set MPS environment variables if not already set
        if not os.getenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO"):
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(cls.MPS.HIGH_WATERMARK_RATIO)
            config_applied["mps_high_watermark"] = cls.MPS.HIGH_WATERMARK_RATIO
        
        if not os.getenv("PYTORCH_ENABLE_MPS_FALLBACK"):
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = str(int(cls.MPS.ENABLE_FALLBACK_DEFAULT))
            config_applied["mps_fallback"] = cls.MPS.ENABLE_FALLBACK_DEFAULT
        
        # Detect hardware capabilities
        hardware_info = cls.Performance.detect_apple_silicon()
        config_applied["hardware"] = hardware_info
        
        # Get optimal device
        optimal_device = cls.Performance.get_optimal_device()
        config_applied["optimal_device"] = str(optimal_device)
        
        return config_applied
    
    
    @classmethod
    def get_apple_silicon_summary(cls) -> str:
        """Generate comprehensive Apple Silicon configuration summary."""
        hardware_info = cls.Performance.detect_apple_silicon()
        memory_info = cls.MPS.get_memory_info()
        ane_info = cls.ANE.get_ane_info()
        
        return f"""
        Apple Silicon Configuration Summary:
        
        Hardware Detection:
        - Platform: {hardware_info.get('platform', 'Unknown')}
        - Architecture: {hardware_info.get('machine', 'Unknown')}
        - Apple Silicon: {hardware_info.get('is_apple_silicon', False)}
        
        MPS Backend:
        - Available: {hardware_info.get('mps_available', False)}
        - Built: {hardware_info.get('mps_built', False)}
        - High Watermark: {cls.MPS.HIGH_WATERMARK_RATIO}
        - Fallback Enabled: {cls.MPS.ENABLE_FALLBACK_DEFAULT}
        
        Memory Configuration:
        - Memory Info: {memory_info}
        - Pressure Threshold: {cls.UnifiedMemory.MEMORY_PRESSURE_THRESHOLD}
        - Swap Prevention: {cls.UnifiedMemory.SWAP_PREVENTION_ENABLED}
        
        ANE Configuration:
        - Target Utilization: {cls.ANE.TARGET_ANE_UTILIZATION}
        - Max Model Size: {cls.ANE.MAX_MODEL_SIZE_MB} MB
        - Target Latency: {cls.ANE.TARGET_LATENCY_MS} ms
        - Compute Precision: {cls.ANE.COMPUTE_PRECISION}
        
        Performance Targets:
        - Target FPS: {cls.Performance.TARGET_FPS}
        - GPU Utilization: {cls.Performance.TARGET_GPU_UTILIZATION}
        - Memory Bandwidth: {cls.Performance.TARGET_MEMORY_BANDWIDTH}
        """