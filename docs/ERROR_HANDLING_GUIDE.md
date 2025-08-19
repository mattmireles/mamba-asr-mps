# Error Handling and State Management Guide for MambaASR

## Overview

This document provides comprehensive error handling and state management patterns for the MambaASR system, designed for AI-first development. Every error condition, recovery strategy, and state transition is documented to enable robust system operation across diverse deployment scenarios.

## Error Handling Philosophy

### Core Principles

1. **Graceful Degradation**: System continues operation with reduced functionality rather than complete failure
2. **Explicit Fallbacks**: All fallback strategies are explicitly documented and implemented
3. **Error Context Preservation**: All errors include sufficient context for debugging and recovery
4. **Platform-Specific Handling**: Apple Silicon hardware characteristics considered in all error scenarios
5. **AI-First Documentation**: Every error pattern includes context for AI systems to understand and extend

### Error Classification Hierarchy

```
System Errors
├── Hardware Errors (Apple Silicon specific)
│   ├── MPS Backend Failures
│   ├── Memory Pressure Events
│   └── Device Unavailability
├── Model Errors
│   ├── Training Failures
│   ├── Export/Conversion Errors
│   └── Inference Runtime Errors
├── Data Errors
│   ├── Input Validation Failures
│   ├── File System Errors
│   └── Format Mismatches
└── Configuration Errors
    ├── Environment Variable Issues
    ├── Invalid Parameter Combinations
    └── Missing Dependencies
```

## Hardware-Specific Error Handling

### MPS Backend Error Patterns

#### Pattern 1: MPS Unavailability

**Error Location**: Device detection in training scripts and configuration
**Error Signature**: `torch.backends.mps.is_available() returns False`

```python
# File: config/apple_silicon_config.py
# Error Handling Implementation

class MPSUnavailabilityHandler:
    """Handles MPS backend unavailability with comprehensive fallback."""
    
    @staticmethod
    def detect_and_handle_mps_unavailability():
        """Detect MPS availability and implement fallback strategy.
        
        Error Conditions:
        1. PyTorch not built with MPS support
        2. macOS version < 12.3 (no MPS support)
        3. Non-Apple Silicon hardware
        4. MPS runtime initialization failure
        
        Recovery Strategy:
        1. Log specific unavailability reason
        2. Fall back to CUDA if available
        3. Fall back to CPU with performance warning
        4. Continue execution with degraded performance
        
        State Transitions:
        Initial → MPS Detection → [Available|Unavailable] → Device Selection → Training
        """
        
        availability_context = {
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available() if torch.backends.mps.is_built() else False,
            "platform": platform.platform(),
            "is_apple_silicon": platform.machine() == "arm64"
        }
        
        if not availability_context["mps_built"]:
            error_context = {
                "error_type": "mps_not_built",
                "resolution": "Install PyTorch with MPS support",
                "fallback": "cpu",
                "performance_impact": "significant"
            }
            return torch.device("cpu"), error_context
        
        if not availability_context["mps_available"]:
            error_context = {
                "error_type": "mps_runtime_unavailable",
                "possible_causes": ["macOS < 12.3", "hardware incompatibility", "system configuration"],
                "fallback": "cpu",
                "performance_impact": "significant"
            }
            return torch.device("cpu"), error_context
        
        # MPS available - verify functionality
        try:
            test_tensor = torch.randn(10, 10, device="mps")
            torch.matmul(test_tensor, test_tensor.T)
            return torch.device("mps"), {"status": "verified"}
        except Exception as e:
            error_context = {
                "error_type": "mps_verification_failed",
                "error": str(e),
                "fallback": "cpu",
                "performance_impact": "significant"
            }
            return torch.device("cpu"), error_context
```

#### Pattern 2: MPS Operation Not Supported

**Error Location**: Training loops, model operations
**Error Signature**: `NotImplementedError` or `RuntimeError` with MPS operations

```python
# File: modules/rnnt_loss_mps.py
# Error Handling Implementation

class MPSOperationErrorHandler:
    """Handles MPS operation failures with automatic CPU fallback."""
    
    @staticmethod
    def safe_mps_operation(operation_fn, *args, fallback_device="cpu", **kwargs):
        """Execute operation with MPS error handling and CPU fallback.
        
        Error Conditions:
        1. NotImplementedError: Operation not supported on MPS
        2. RuntimeError: MPS runtime error during operation
        3. Memory errors: Insufficient memory for operation
        
        Recovery Strategy:
        1. Capture original error context
        2. Move tensors to CPU
        3. Execute operation on CPU
        4. Move result back to original device
        5. Log fallback occurrence for monitoring
        
        State Management:
        Device State: MPS → CPU (temporary) → MPS
        Tensor State: Device placement preserved across fallback
        Error State: Captured and logged for analysis
        """
        
        try:
            return operation_fn(*args, **kwargs)
        except (NotImplementedError, RuntimeError) as e:
            if "mps" in str(e).lower() or "metal" in str(e).lower():
                # Log MPS-specific error
                error_context = {
                    "error_type": "mps_operation_unsupported",
                    "operation": operation_fn.__name__,
                    "error_message": str(e),
                    "fallback_applied": True,
                    "timestamp": time.time()
                }
                
                # Apply CPU fallback
                cpu_args = [arg.cpu() if hasattr(arg, 'cpu') else arg for arg in args]
                cpu_kwargs = {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in kwargs.items()}
                
                result = operation_fn(*cpu_args, **cpu_kwargs)
                
                # Return result to original device if possible
                if hasattr(result, 'to') and len(args) > 0 and hasattr(args[0], 'device'):
                    result = result.to(args[0].device)
                
                return result, error_context
            else:
                # Re-raise non-MPS errors
                raise
```

### Memory Management Error Patterns

#### Pattern 3: Memory Pressure on Apple Silicon

**Error Location**: Training loops, large model operations
**Error Signature**: System memory pressure, swapping behavior

```python
# File: config/apple_silicon_config.py
# Memory Management Implementation

class UnifiedMemoryManager:
    """Manages unified memory architecture constraints and pressure."""
    
    def __init__(self):
        self.memory_threshold = AppleSiliconConfig.UnifiedMemory.MEMORY_PRESSURE_THRESHOLD
        self.current_pressure_level = 0
        self.pressure_history = []
    
    def monitor_memory_pressure(self):
        """Monitor system memory pressure with graduated response.
        
        Error Conditions:
        1. Memory usage > 85% of system RAM
        2. Swap file usage detected
        3. System memory warnings from macOS
        
        Recovery Strategy:
        1. Level 1 (70-80%): Log warning, continue operation
        2. Level 2 (80-90%): Clear caches, reduce batch size
        3. Level 3 (90%+): Force garbage collection, emergency fallback
        
        State Transitions:
        Normal → Warning → Critical → Emergency → Recovery
        """
        
        memory_info = self.get_system_memory_info()
        pressure_level = self.calculate_pressure_level(memory_info)
        
        if pressure_level > self.current_pressure_level:
            self.handle_pressure_increase(pressure_level, memory_info)
        elif pressure_level < self.current_pressure_level:
            self.handle_pressure_decrease(pressure_level, memory_info)
        
        self.current_pressure_level = pressure_level
        self.pressure_history.append({
            "timestamp": time.time(),
            "pressure_level": pressure_level,
            "memory_info": memory_info
        })
    
    def handle_pressure_increase(self, pressure_level, memory_info):
        """Handle increasing memory pressure with graduated response."""
        
        if pressure_level == 1:  # Warning level
            self.log_memory_warning(memory_info)
        
        elif pressure_level == 2:  # Critical level
            self.apply_memory_conservation_measures()
        
        elif pressure_level >= 3:  # Emergency level
            self.apply_emergency_memory_measures()
    
    def apply_memory_conservation_measures(self):
        """Apply memory conservation measures for critical pressure."""
        
        # Clear PyTorch caches
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Log conservation measures applied
        conservation_log = {
            "action": "memory_conservation_applied",
            "measures": ["cache_clearing", "garbage_collection"],
            "timestamp": time.time()
        }
        
        return conservation_log
    
    def apply_emergency_memory_measures(self):
        """Apply emergency memory measures for severe pressure."""
        
        emergency_actions = []
        
        # Apply conservation measures first
        conservation_log = self.apply_memory_conservation_measures()
        emergency_actions.append("conservation_measures")
        
        # Reduce operation complexity
        emergency_actions.append("complexity_reduction")
        
        # Signal emergency state to training loops
        self.signal_emergency_state()
        emergency_actions.append("emergency_signaling")
        
        emergency_log = {
            "action": "emergency_memory_measures",
            "actions_taken": emergency_actions,
            "timestamp": time.time(),
            "conservation_details": conservation_log
        }
        
        return emergency_log
```

## Model Error Handling Patterns

### Training Error Patterns

#### Pattern 4: RNN-T Loss Computation Failures

**Error Location**: `modules/rnnt_loss_mps.py`
**Error Signature**: Various backend-specific failures

```python
# File: modules/rnnt_loss_mps.py
# Training Error Handling Implementation

class RNNTLossErrorHandler:
    """Comprehensive error handling for RNN-T loss computation."""
    
    def __init__(self):
        self.failure_history = {}
        self.backend_reliability = {
            "torchaudio": {"successes": 0, "failures": 0},
            "warp_rnnt": {"successes": 0, "failures": 0},
            "cpu_grad": {"successes": 0, "failures": 0}
        }
    
    def compute_loss_with_error_handling(self, logits, targets, input_lengths, target_lengths, max_align=None):
        """Compute RNN-T loss with comprehensive error handling and fallback.
        
        Error Conditions:
        1. Backend unavailable: ImportError during backend selection
        2. Shape mismatch: Input tensors have incompatible shapes
        3. Memory overflow: Alignment matrix T*U exceeds memory limits
        4. Numerical instability: NaN or Inf values in computation
        5. Device mismatch: Tensors on incompatible devices
        
        Recovery Strategy:
        1. Attempt primary backend (torchaudio)
        2. Fall back to secondary backend (warp_rnnt)
        3. Fall back to CPU gradient computation
        4. Emergency CTC fallback if all RNN-T backends fail
        
        State Management:
        Backend State: Track reliability of each backend
        Error State: Log all failures for analysis
        Training State: Maintain training progress regardless of backend
        """
        
        # Validate inputs before computation
        validation_result = self.validate_inputs(logits, targets, input_lengths, target_lengths, max_align)
        if validation_result["status"] == "error":
            return self.handle_validation_error(validation_result)
        
        # Attempt computation with error handling
        backends_to_try = ["torchaudio", "warp_rnnt", "cpu_grad"]
        
        for backend_name in backends_to_try:
            try:
                loss, gradients, actual_backend = self.try_backend(
                    backend_name, logits, targets, input_lengths, target_lengths, max_align
                )
                
                # Record success
                self.record_backend_success(actual_backend)
                return loss, gradients, actual_backend
                
            except Exception as e:
                # Record failure and continue to next backend
                failure_info = self.record_backend_failure(backend_name, e, {
                    "logits_shape": logits.shape,
                    "targets_shape": targets.shape,
                    "input_lengths": input_lengths.tolist(),
                    "target_lengths": target_lengths.tolist(),
                    "max_align": max_align
                })
                
                # Log detailed failure information
                print(f"Backend {backend_name} failed: {failure_info}")
                continue
        
        # All RNN-T backends failed - emergency CTC fallback
        return self.emergency_ctc_fallback(logits, targets, input_lengths, target_lengths)
    
    def validate_inputs(self, logits, targets, input_lengths, target_lengths, max_align):
        """Validate inputs before RNN-T computation.
        
        Validation Checks:
        1. Tensor shapes compatibility
        2. Device consistency
        3. Memory requirements estimation
        4. Numerical validity (no NaN/Inf)
        
        Error Prevention:
        - Catch shape mismatches before computation
        - Estimate memory requirements
        - Validate tensor contents
        """
        
        validation_errors = []
        
        # Shape validation
        batch_size = logits.shape[0]
        if targets.shape[0] != batch_size:
            validation_errors.append(f"Batch size mismatch: logits {batch_size}, targets {targets.shape[0]}")
        
        if input_lengths.shape[0] != batch_size:
            validation_errors.append(f"Input lengths batch size mismatch: {input_lengths.shape[0]} vs {batch_size}")
        
        if target_lengths.shape[0] != batch_size:
            validation_errors.append(f"Target lengths batch size mismatch: {target_lengths.shape[0]} vs {batch_size}")
        
        # Device validation
        if not all(tensor.device == logits.device for tensor in [targets, input_lengths, target_lengths]):
            validation_errors.append("Device mismatch between input tensors")
        
        # Memory estimation
        max_T = int(input_lengths.max().item())
        max_U = int(target_lengths.max().item())
        estimated_memory = batch_size * max_T * max_U * logits.shape[-1] * 4  # 4 bytes per float32
        
        if max_align and max_T * max_U > max_align:
            validation_errors.append(f"Alignment constraint violated: T*U={max_T * max_U} > max_align={max_align}")
        
        # Numerical validation
        if torch.isnan(logits).any():
            validation_errors.append("NaN values detected in logits")
        
        if torch.isinf(logits).any():
            validation_errors.append("Inf values detected in logits")
        
        if validation_errors:
            return {
                "status": "error",
                "errors": validation_errors,
                "estimated_memory_mb": estimated_memory / (1024 * 1024),
                "shapes": {
                    "logits": logits.shape,
                    "targets": targets.shape,
                    "input_lengths": input_lengths.shape,
                    "target_lengths": target_lengths.shape
                }
            }
        
        return {
            "status": "valid",
            "estimated_memory_mb": estimated_memory / (1024 * 1024),
            "max_T": max_T,
            "max_U": max_U
        }
    
    def emergency_ctc_fallback(self, logits, targets, input_lengths, target_lengths):
        """Emergency CTC fallback when all RNN-T backends fail.
        
        Fallback Strategy:
        1. Convert RNN-T inputs to CTC format
        2. Use PyTorch native CTC loss
        3. Log emergency fallback occurrence
        4. Return approximated loss for training continuation
        
        State Impact:
        - Training continues with CTC approximation
        - Emergency state logged for monitoring
        - Backend reliability updated
        """
        
        emergency_log = {
            "action": "emergency_ctc_fallback",
            "reason": "all_rnnt_backends_failed",
            "timestamp": time.time(),
            "input_stats": {
                "batch_size": logits.shape[0],
                "max_T": int(input_lengths.max().item()),
                "max_U": int(target_lengths.max().item())
            }
        }
        
        try:
            # Convert to CTC format (remove predictor dimension)
            ctc_logits = logits.mean(dim=2)  # Average over predictor dimension
            ctc_log_probs = torch.log_softmax(ctc_logits, dim=-1)
            
            # Remove leading blank from targets for CTC
            ctc_targets = targets[:, 1:]  # Skip blank token
            ctc_target_lengths = target_lengths - 1  # Adjust for removed blank
            
            # Compute CTC loss
            ctc_loss = torch.nn.functional.ctc_loss(
                ctc_log_probs.transpose(0, 1),  # CTC expects (T, B, V)
                ctc_targets,
                input_lengths,
                ctc_target_lengths,
                blank=0,
                reduction='mean'
            )
            
            emergency_log["status"] = "success"
            emergency_log["ctc_loss"] = ctc_loss.item()
            
            return ctc_loss, None, "emergency_ctc"
            
        except Exception as e:
            emergency_log["status"] = "failed"
            emergency_log["error"] = str(e)
            
            # Return dummy loss to prevent training crash
            dummy_loss = torch.tensor(1.0, device=logits.device, requires_grad=True)
            return dummy_loss, None, "emergency_dummy"
```

### Export and Conversion Error Patterns

#### Pattern 5: Core ML Export Failures

**Error Location**: `scripts/export_coreml.py`
**Error Signature**: Various Core ML conversion errors

```python
# File: scripts/export_coreml.py
# Export Error Handling Implementation

class CoreMLExportErrorHandler:
    """Handles Core ML export errors with detailed diagnostics."""
    
    def __init__(self):
        self.export_history = []
        self.known_issues = self.load_known_issues()
    
    def export_with_error_handling(self, model, example_input, output_path, config=None):
        """Export PyTorch model to Core ML with comprehensive error handling.
        
        Error Conditions:
        1. Unsupported operations: PyTorch ops not available in Core ML
        2. Shape incompatibility: Dynamic shapes not handled correctly
        3. Quantization failures: QAT models with unsupported precision
        4. Memory constraints: Model too large for Core ML compilation
        5. Version incompatibility: Core ML tools version issues
        
        Recovery Strategy:
        1. Diagnose specific error type
        2. Apply known workarounds if available
        3. Suggest alternative export strategies
        4. Generate detailed error report for debugging
        
        State Management:
        Export State: Track all export attempts and outcomes
        Model State: Preserve original model unchanged
        Configuration State: Document successful export configurations
        """
        
        export_attempt = {
            "timestamp": time.time(),
            "model_type": type(model).__name__,
            "output_path": output_path,
            "config": config,
            "status": "in_progress"
        }
        
        try:
            # Pre-export validation
            validation_result = self.validate_export_preconditions(model, example_input)
            if validation_result["status"] == "error":
                return self.handle_precondition_error(validation_result, export_attempt)
            
            # Attempt Core ML conversion
            traced_model = torch.jit.trace(model, example_input)
            
            # Configure conversion parameters
            conversion_config = self.prepare_conversion_config(config, model, example_input)
            
            # Execute conversion with error capture
            coreml_model = ct.convert(
                traced_model,
                **conversion_config
            )
            
            # Post-conversion validation
            validation_result = self.validate_exported_model(coreml_model, example_input)
            if validation_result["status"] == "error":
                return self.handle_post_conversion_error(validation_result, export_attempt)
            
            # Save model
            coreml_model.save(output_path)
            
            export_attempt.update({
                "status": "success",
                "output_size_mb": self.get_file_size_mb(output_path),
                "validation_result": validation_result
            })
            
            self.export_history.append(export_attempt)
            return {"status": "success", "model": coreml_model, "export_info": export_attempt}
            
        except Exception as e:
            # Detailed error analysis
            error_analysis = self.analyze_export_error(e, model, example_input, config)
            
            export_attempt.update({
                "status": "failed",
                "error": str(e),
                "error_analysis": error_analysis
            })
            
            self.export_history.append(export_attempt)
            
            # Attempt recovery strategies
            recovery_result = self.attempt_error_recovery(e, model, example_input, output_path, config)
            
            if recovery_result["status"] == "success":
                return recovery_result
            else:
                return {"status": "failed", "error_info": export_attempt, "recovery_attempts": recovery_result}
    
    def analyze_export_error(self, error, model, example_input, config):
        """Analyze Core ML export error and provide diagnostic information.
        
        Error Analysis Categories:
        1. Unsupported Operations: Identify specific PyTorch ops causing issues
        2. Shape Problems: Dynamic shape handling failures
        3. Quantization Issues: QAT/quantization-related problems
        4. Memory Issues: Model size or complexity problems
        5. Configuration Issues: Invalid conversion parameters
        """
        
        error_str = str(error)
        error_type = type(error).__name__
        
        analysis = {
            "error_type": error_type,
            "error_message": error_str,
            "analysis_timestamp": time.time(),
            "model_info": {
                "type": type(model).__name__,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "has_quantization": self.detect_quantization(model)
            }
        }
        
        # Categorize error
        if "unsupported" in error_str.lower():
            analysis["category"] = "unsupported_operations"
            analysis["suggested_actions"] = [
                "Review model architecture for Core ML compatibility",
                "Consider replacing unsupported operations",
                "Check Core ML supported operations documentation"
            ]
        
        elif "shape" in error_str.lower() or "dimension" in error_str.lower():
            analysis["category"] = "shape_incompatibility"
            analysis["suggested_actions"] = [
                "Verify input shape specifications",
                "Consider using fixed input shapes",
                "Check dynamic shape handling in Core ML"
            ]
        
        elif "quantization" in error_str.lower() or "precision" in error_str.lower():
            analysis["category"] = "quantization_issues"
            analysis["suggested_actions"] = [
                "Export without quantization first",
                "Verify QAT model compatibility",
                "Consider post-training quantization"
            ]
        
        elif "memory" in error_str.lower() or "size" in error_str.lower():
            analysis["category"] = "memory_constraints"
            analysis["suggested_actions"] = [
                "Reduce model size",
                "Consider model pruning",
                "Export on machine with more memory"
            ]
        
        else:
            analysis["category"] = "unknown"
            analysis["suggested_actions"] = [
                "Check Core ML tools version compatibility",
                "Verify PyTorch model correctness",
                "Consult Core ML documentation"
            ]
        
        return analysis
```

## State Management Patterns

### Training State Lifecycle

```python
# File: docs/state_management_patterns.py
# Training State Management Implementation

class TrainingStateManager:
    """Manages training state lifecycle with comprehensive error handling."""
    
    def __init__(self, checkpoint_dir="checkpoints/"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.state = {
            "training_phase": "initialization",
            "current_epoch": 0,
            "current_step": 0,
            "best_metrics": {},
            "device_info": {},
            "error_history": [],
            "recovery_attempts": []
        }
    
    def transition_state(self, new_phase, context=None):
        """Manage state transitions with validation and logging.
        
        State Transition Validation:
        1. Verify transition is valid from current state
        2. Ensure required conditions are met
        3. Log transition for debugging
        4. Update dependent components
        
        State Persistence:
        - All state transitions logged
        - Critical states checkpointed
        - Recovery information maintained
        """
        
        valid_transitions = {
            "initialization": ["device_setup", "error"],
            "device_setup": ["data_loading", "error"],
            "data_loading": ["model_creation", "error"],
            "model_creation": ["training", "error"],
            "training": ["validation", "completed", "error", "paused"],
            "validation": ["training", "completed", "error"],
            "paused": ["training", "completed", "error"],
            "error": ["recovery", "terminated"],
            "recovery": ["training", "error", "terminated"],
            "completed": ["terminated"],
            "terminated": []
        }
        
        current_phase = self.state["training_phase"]
        
        if new_phase not in valid_transitions.get(current_phase, []):
            raise ValueError(f"Invalid state transition: {current_phase} -> {new_phase}")
        
        # Log transition
        transition_log = {
            "timestamp": time.time(),
            "from_phase": current_phase,
            "to_phase": new_phase,
            "context": context,
            "epoch": self.state["current_epoch"],
            "step": self.state["current_step"]
        }
        
        self.state["training_phase"] = new_phase
        self.state.setdefault("transition_history", []).append(transition_log)
        
        # Handle phase-specific actions
        self.handle_phase_transition(new_phase, context)
        
        return transition_log
    
    def handle_phase_transition(self, phase, context):
        """Handle phase-specific transition actions."""
        
        if phase == "error":
            self.handle_error_transition(context)
        elif phase == "recovery":
            self.handle_recovery_transition(context)
        elif phase == "completed":
            self.handle_completion_transition(context)
        elif phase == "terminated":
            self.handle_termination_transition(context)
    
    def handle_error_transition(self, error_context):
        """Handle transition to error state with recovery preparation."""
        
        error_record = {
            "timestamp": time.time(),
            "error_type": error_context.get("error_type", "unknown"),
            "error_message": error_context.get("error_message", ""),
            "training_context": {
                "epoch": self.state["current_epoch"],
                "step": self.state["current_step"],
                "phase": self.state.get("previous_phase", "unknown")
            },
            "recovery_options": self.identify_recovery_options(error_context)
        }
        
        self.state["error_history"].append(error_record)
        
        # Prepare for potential recovery
        self.prepare_recovery_checkpoint(error_context)
    
    def identify_recovery_options(self, error_context):
        """Identify available recovery options based on error type."""
        
        error_type = error_context.get("error_type", "unknown")
        
        recovery_options = {
            "mps_error": [
                "fallback_to_cpu",
                "reduce_batch_size",
                "clear_mps_cache"
            ],
            "memory_error": [
                "reduce_batch_size",
                "clear_caches",
                "gradient_checkpointing"
            ],
            "rnnt_backend_error": [
                "switch_backend",
                "cpu_grad_fallback",
                "ctc_fallback"
            ],
            "data_error": [
                "skip_batch",
                "reload_dataset",
                "validate_data"
            ]
        }
        
        return recovery_options.get(error_type, ["restart_training"])
```

## Configuration Error Handling

### Environment Variable Validation

```python
# File: config/environment_config.py
# Enhanced Error Handling for Configuration

class ConfigurationErrorHandler:
    """Handles configuration errors with detailed diagnostics."""
    
    @classmethod
    def validate_configuration_with_error_handling(cls):
        """Validate entire configuration with comprehensive error reporting.
        
        Validation Categories:
        1. Environment Variable Format: Type and range validation
        2. Cross-Parameter Consistency: Parameter combination validation
        3. Hardware Compatibility: Configuration vs. hardware capabilities
        4. Dependency Availability: Required software/hardware availability
        
        Error Recovery:
        1. Invalid values: Use defaults with warnings
        2. Incompatible combinations: Suggest valid alternatives
        3. Missing dependencies: Provide installation guidance
        """
        
        validation_results = {
            "status": "pending",
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "applied_fixes": []
        }
        
        # Validate individual environment variables
        for var_name, var_config in cls.ENVIRONMENT_VARIABLES.items():
            var_result = cls.validate_single_variable(var_name, var_config)
            
            if var_result["status"] == "error":
                validation_results["errors"].extend(var_result["errors"])
                
                # Apply automatic fixes where possible
                fix_applied = cls.apply_automatic_fix(var_name, var_result)
                if fix_applied:
                    validation_results["applied_fixes"].append(fix_applied)
            
            elif var_result["status"] == "warning":
                validation_results["warnings"].extend(var_result["warnings"])
        
        # Cross-parameter validation
        cross_validation = cls.validate_parameter_combinations()
        validation_results["errors"].extend(cross_validation.get("errors", []))
        validation_results["warnings"].extend(cross_validation.get("warnings", []))
        
        # Hardware compatibility validation
        hardware_validation = cls.validate_hardware_compatibility()
        validation_results["errors"].extend(hardware_validation.get("errors", []))
        validation_results["recommendations"].extend(hardware_validation.get("recommendations", []))
        
        # Final status determination
        if validation_results["errors"]:
            validation_results["status"] = "error"
        elif validation_results["warnings"]:
            validation_results["status"] = "warning"
        else:
            validation_results["status"] = "success"
        
        return validation_results
    
    @classmethod
    def generate_error_report(cls, validation_results):
        """Generate comprehensive error report for configuration issues."""
        
        report_lines = [
            "# MambaASR Configuration Validation Report",
            f"**Status:** {validation_results['status'].upper()}",
            f"**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        if validation_results["errors"]:
            report_lines.extend([
                "## Errors (Must Fix)",
                ""
            ])
            for i, error in enumerate(validation_results["errors"], 1):
                report_lines.extend([
                    f"### Error {i}: {error['type']}",
                    f"**Variable:** {error.get('variable', 'N/A')}",
                    f"**Issue:** {error['message']}",
                    f"**Resolution:** {error.get('resolution', 'Manual intervention required')}",
                    ""
                ])
        
        if validation_results["warnings"]:
            report_lines.extend([
                "## Warnings (Recommended Fixes)",
                ""
            ])
            for i, warning in enumerate(validation_results["warnings"], 1):
                report_lines.extend([
                    f"### Warning {i}: {warning['type']}",
                    f"**Variable:** {warning.get('variable', 'N/A')}",
                    f"**Issue:** {warning['message']}",
                    f"**Recommendation:** {warning.get('recommendation', 'Review configuration')}",
                    ""
                ])
        
        if validation_results["applied_fixes"]:
            report_lines.extend([
                "## Automatic Fixes Applied",
                ""
            ])
            for fix in validation_results["applied_fixes"]:
                report_lines.extend([
                    f"- **{fix['variable']}:** {fix['action']} (was: {fix['old_value']}, now: {fix['new_value']})",
                    ""
                ])
        
        if validation_results["recommendations"]:
            report_lines.extend([
                "## Optimization Recommendations",
                ""
            ])
            for rec in validation_results["recommendations"]:
                report_lines.append(f"- {rec}")
        
        return "\\n".join(report_lines)
```

This comprehensive error handling and state management guide provides AI developers with complete context for understanding, debugging, and extending error handling throughout the MambaASR system. Every error condition includes specific recovery strategies and state management patterns optimized for Apple Silicon deployment scenarios.