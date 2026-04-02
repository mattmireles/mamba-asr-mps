"""
Environment Variable Configuration Management for MambaASR

This module provides standardized environment variable handling for MambaASR
configuration override and deployment customization. It enables runtime
configuration adjustment without code modification, supporting development,
testing, and production deployment scenarios.

Environment Variable Strategy:
- Hierarchical naming: Component prefix + parameter name (MAMBA_BATCH_SIZE)
- Type safety: Automatic type conversion with validation
- Default preservation: Environment variables override defaults without changing them
- Documentation: Each variable documented with purpose and valid ranges
- Error handling: Graceful degradation when invalid values provided

Integration Patterns:
- Development: Override parameters for experimentation and debugging
- CI/CD: Automated testing with different configuration combinations
- Production: Deployment-specific optimization without code changes
- Research: Parameter sweeps and hyperparameter optimization

Cross-Component Coverage:
- Training: Batch size, learning rate, optimization parameters
- Model: Architecture dimensions, vocabulary size, component configuration
- Hardware: Apple Silicon optimization, memory management, device selection
- Evaluation: Thresholds, metrics, output format configuration
- Deployment: Core ML settings, ANE optimization, performance targets

AI-First Design Principles:
- Comprehensive documentation for each environment variable
- Clear mapping between environment variables and configuration constants
- Type safety and validation to prevent runtime errors
- Integration examples for common deployment scenarios
"""

import os
import logging
from typing import Dict, Any, Optional, Union, Type, Callable


class EnvironmentConfig:
    """Environment variable configuration management for MambaASR system.
    
    This class provides standardized handling of environment variable overrides
    for all MambaASR configuration parameters. It ensures type safety, provides
    validation, and maintains comprehensive documentation for deployment teams.
    """
    
    # Environment variable definitions with metadata
    ENVIRONMENT_VARIABLES = {
        # Training Configuration
        "MAMBA_BATCH_SIZE": {
            "type": int,
            "default": 2,
            "min_value": 1,
            "max_value": 16,
            "description": "Training batch size optimized for Apple Silicon memory constraints",
            "config_path": "Training.DEFAULT_BATCH_SIZE",
            "example": "export MAMBA_BATCH_SIZE=4"
        },
        "MAMBA_LEARNING_RATE": {
            "type": float,
            "default": 3e-4,
            "min_value": 1e-6,
            "max_value": 1e-2,
            "description": "AdamW learning rate for RNN-T training optimization",
            "config_path": "Training.DEFAULT_LEARNING_RATE",
            "example": "export MAMBA_LEARNING_RATE=1e-4"
        },
        "MAMBA_EPOCHS": {
            "type": int,
            "default": 1,
            "min_value": 1,
            "max_value": 1000,
            "description": "Number of training epochs for model convergence",
            "config_path": "Training.DEFAULT_EPOCHS",
            "example": "export MAMBA_EPOCHS=100"
        },
        "MAMBA_GRAD_CLIP": {
            "type": float,
            "default": 1.0,
            "min_value": 0.1,
            "max_value": 10.0,
            "description": "Global gradient norm clipping for training stability",
            "config_path": "Training.DEFAULT_GRAD_CLIP",
            "example": "export MAMBA_GRAD_CLIP=0.5"
        },
        
        # Model Architecture Configuration
        "MAMBA_D_MODEL": {
            "type": int,
            "default": 256,
            "min_value": 64,
            "max_value": 2048,
            "description": "Model dimension for Mamba encoder and other components",
            "config_path": "Model.DEFAULT_D_MODEL",
            "example": "export MAMBA_D_MODEL=512"
        },
        "MAMBA_N_BLOCKS": {
            "type": int,
            "default": 4,
            "min_value": 1,
            "max_value": 16,
            "description": "Number of Mamba blocks in encoder architecture",
            "config_path": "Model.DEFAULT_N_BLOCKS",
            "example": "export MAMBA_N_BLOCKS=6"
        },
        "MAMBA_JOINT_DIM": {
            "type": int,
            "default": 320,
            "min_value": 128,
            "max_value": 1024,
            "description": "RNN-T joiner dimension for acoustic-linguistic fusion",
            "config_path": "Model.DEFAULT_JOINT_DIM",
            "example": "export MAMBA_JOINT_DIM=512"
        },
        "MAMBA_VOCAB_SIZE": {
            "type": int,
            "default": 1024,
            "min_value": 29,
            "max_value": 50000,
            "description": "Vocabulary size (29 for character tokenizer, larger for subwords)",
            "config_path": "Model.DEFAULT_VOCAB_SIZE",
            "example": "export MAMBA_VOCAB_SIZE=29"
        },
        
        # RNN-T Loss Configuration
        "RNNT_MAX_ALIGN": {
            "type": int,
            "default": 60000,
            "min_value": 1000,
            "max_value": 1000000,
            "description": "Maximum T*U alignment constraint for memory management",
            "config_path": "RNNTLoss.DEFAULT_MAX_ALIGNMENT",
            "example": "export RNNT_MAX_ALIGN=80000"
        },
        "RNNT_CLAMP_VALUE": {
            "type": float,
            "default": -1.0,
            "min_value": -1.0,
            "max_value": 100.0,
            "description": "Loss clamping value (-1 disables clamping)",
            "config_path": "RNNTLoss.DEFAULT_CLAMP_VALUE",
            "example": "export RNNT_CLAMP_VALUE=10.0"
        },
        
        # Apple Silicon MPS Configuration
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": {
            "type": float,
            "default": 0.8,
            "min_value": 0.5,
            "max_value": 1.0,
            "description": "MPS memory allocation threshold for unified memory pressure management",
            "config_path": "AppleSilicon.MPS.HIGH_WATERMARK_RATIO",
            "example": "export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7"
        },
        "PYTORCH_ENABLE_MPS_FALLBACK": {
            "type": bool,
            "default": True,
            "description": "Enable CPU fallback for unsupported MPS operations",
            "config_path": "AppleSilicon.MPS.ENABLE_FALLBACK_DEFAULT",
            "example": "export PYTORCH_ENABLE_MPS_FALLBACK=1"
        },
        
        # Evaluation Configuration
        "MAMBA_CER_THRESHOLD": {
            "type": float,
            "default": 0.6,
            "min_value": 0.0,
            "max_value": 1.0,
            "description": "Character Error Rate threshold for evaluation gating",
            "config_path": "TextProcessing.DEFAULT_CER_THRESHOLD",
            "example": "export MAMBA_CER_THRESHOLD=0.3"
        },
        "MAMBA_WER_THRESHOLD": {
            "type": float,
            "default": 0.3,
            "min_value": 0.0,
            "max_value": 1.0,
            "description": "Word Error Rate threshold for evaluation gating",
            "config_path": "TextProcessing.DEFAULT_WER_THRESHOLD",
            "example": "export MAMBA_WER_THRESHOLD=0.2"
        },
        
        # Performance and Profiling
        "MAMBA_DISABLE_RECORD_FUNCTION": {
            "type": bool,
            "default": False,
            "description": "Disable PyTorch record_function profiling for production",
            "config_path": "Performance.DISABLE_PROFILING",
            "example": "export MAMBA_DISABLE_RECORD_FUNCTION=1"
        },
        
        # Core ML Export Configuration
        "MAMBA_CHUNK_DEFAULT": {
            "type": int,
            "default": 256,
            "min_value": 64,
            "max_value": 1024,
            "description": "Default chunk size for Core ML model export",
            "config_path": "CoreML.DEFAULT_CHUNK_SIZE",
            "example": "export MAMBA_CHUNK_DEFAULT=512"
        },
        "MAMBA_COMPUTE_DEFAULT": {
            "type": str,
            "default": "cpu",
            "valid_values": ["all", "cpu", "cpu-gpu", "ane"],
            "description": "Default compute configuration for Core ML inference",
            "config_path": "CoreML.DEFAULT_COMPUTE_MODE",
            "example": "export MAMBA_COMPUTE_DEFAULT=all"
        }
    }
    
    
    @classmethod
    def get_environment_value(
        cls, 
        variable_name: str, 
        default: Any = None,
        validate: bool = True
    ) -> Any:
        """Get environment variable value with type conversion and validation.
        
        Args:
            variable_name: Environment variable name (e.g., 'MAMBA_BATCH_SIZE')
            default: Default value if variable not set (overrides metadata default)
            validate: Whether to perform range/value validation
            
        Returns:
            Parsed and validated environment variable value or default
            
        Raises:
            ValueError: If validation fails for provided value
            TypeError: If type conversion fails
        """
        if variable_name not in cls.ENVIRONMENT_VARIABLES:
            logging.warning(f"Unknown environment variable: {variable_name}")
            return os.getenv(variable_name, default)
        
        var_config = cls.ENVIRONMENT_VARIABLES[variable_name]
        raw_value = os.getenv(variable_name)
        
        # Use provided default or metadata default
        default_value = default if default is not None else var_config["default"]
        
        if raw_value is None:
            return default_value
        
        # Type conversion
        target_type = var_config["type"]
        try:
            if target_type == bool:
                # Handle boolean environment variables (0/1, true/false, etc.)
                converted_value = raw_value.lower() in ('1', 'true', 'yes', 'on')
            elif target_type == str:
                converted_value = raw_value
            else:
                converted_value = target_type(raw_value)
        except (ValueError, TypeError) as e:
            logging.error(f"Invalid type for {variable_name}: {raw_value}. Expected {target_type.__name__}")
            return default_value
        
        # Validation
        if validate:
            validation_error = cls._validate_value(variable_name, converted_value, var_config)
            if validation_error:
                logging.error(f"Validation failed for {variable_name}: {validation_error}")
                return default_value
        
        return converted_value
    
    
    @classmethod
    def _validate_value(cls, variable_name: str, value: Any, config: Dict[str, Any]) -> Optional[str]:
        """Validate environment variable value against configuration constraints.
        
        Args:
            variable_name: Environment variable name
            value: Parsed value to validate
            config: Variable configuration metadata
            
        Returns:
            Error message if validation fails, None if valid
        """
        # Range validation for numeric types
        if "min_value" in config and value < config["min_value"]:
            return f"Value {value} below minimum {config['min_value']}"
        
        if "max_value" in config and value > config["max_value"]:
            return f"Value {value} above maximum {config['max_value']}"
        
        # Valid values constraint for strings
        if "valid_values" in config and value not in config["valid_values"]:
            return f"Value '{value}' not in valid values: {config['valid_values']}"
        
        return None
    
    
    @classmethod
    def get_all_environment_overrides(cls) -> Dict[str, Any]:
        """Get all current environment variable overrides with validation.
        
        Returns:
            Dictionary mapping variable names to parsed/validated values
        """
        overrides = {}
        
        for var_name in cls.ENVIRONMENT_VARIABLES:
            value = cls.get_environment_value(var_name)
            default = cls.ENVIRONMENT_VARIABLES[var_name]["default"]
            
            # Only include if different from default
            if value != default:
                overrides[var_name] = value
        
        return overrides
    
    
    @classmethod
    def set_development_defaults(cls) -> Dict[str, str]:
        """Set development-friendly environment variable defaults.
        
        Returns:
            Dictionary of environment variables that were set
        """
        development_config = {
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "MAMBA_DISABLE_RECORD_FUNCTION": "0",
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.9"
        }
        
        set_variables = {}
        for var_name, value in development_config.items():
            if not os.getenv(var_name):
                os.environ[var_name] = value
                set_variables[var_name] = value
        
        return set_variables
    
    
    @classmethod
    def set_production_defaults(cls) -> Dict[str, str]:
        """Set production-optimized environment variable defaults.
        
        Returns:
            Dictionary of environment variables that were set
        """
        production_config = {
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            "MAMBA_DISABLE_RECORD_FUNCTION": "1", 
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.8"
        }
        
        set_variables = {}
        for var_name, value in production_config.items():
            if not os.getenv(var_name):
                os.environ[var_name] = value
                set_variables[var_name] = value
        
        return set_variables
    
    
    @classmethod
    def generate_environment_documentation(cls) -> str:
        """Generate comprehensive environment variable documentation.
        
        Returns:
            Formatted documentation string for all environment variables
        """
        doc_lines = [
            "# MambaASR Environment Variable Configuration",
            "",
            "This document describes all environment variables that can be used to",
            "configure MambaASR behavior without modifying code. All variables are",
            "optional and have sensible defaults for Apple Silicon optimization.",
            ""
        ]
        
        # Group variables by category
        categories = {
            "Training": ["MAMBA_BATCH_SIZE", "MAMBA_LEARNING_RATE", "MAMBA_EPOCHS", "MAMBA_GRAD_CLIP"],
            "Model Architecture": ["MAMBA_D_MODEL", "MAMBA_N_BLOCKS", "MAMBA_JOINT_DIM", "MAMBA_VOCAB_SIZE"],
            "RNN-T Loss": ["RNNT_MAX_ALIGN", "RNNT_CLAMP_VALUE"],
            "Apple Silicon": ["PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_ENABLE_MPS_FALLBACK"],
            "Evaluation": ["MAMBA_CER_THRESHOLD", "MAMBA_WER_THRESHOLD"],
            "Performance": ["MAMBA_DISABLE_RECORD_FUNCTION"],
            "Core ML": ["MAMBA_CHUNK_DEFAULT", "MAMBA_COMPUTE_DEFAULT"]
        }
        
        for category, var_names in categories.items():
            doc_lines.extend([f"## {category} Configuration", ""])
            
            for var_name in var_names:
                if var_name in cls.ENVIRONMENT_VARIABLES:
                    var_config = cls.ENVIRONMENT_VARIABLES[var_name]
                    
                    doc_lines.extend([
                        f"### {var_name}",
                        f"**Description:** {var_config['description']}",
                        f"**Type:** {var_config['type'].__name__}",
                        f"**Default:** {var_config['default']}",
                    ])
                    
                    if "min_value" in var_config:
                        doc_lines.append(f"**Range:** {var_config['min_value']} - {var_config['max_value']}")
                    
                    if "valid_values" in var_config:
                        doc_lines.append(f"**Valid Values:** {', '.join(var_config['valid_values'])}")
                    
                    doc_lines.extend([
                        f"**Example:** `{var_config['example']}`",
                        ""
                    ])
        
        return "\\n".join(doc_lines)
    
    
    @classmethod
    def get_environment_summary(cls) -> str:
        """Generate environment configuration summary for logging/debugging.
        
        Returns:
            Formatted summary of current environment configuration
        """
        overrides = cls.get_all_environment_overrides()
        
        if not overrides:
            return "Environment Configuration: Using all default values"
        
        summary_lines = [
            "Environment Configuration Overrides:",
            ""
        ]
        
        for var_name, value in overrides.items():
            config = cls.ENVIRONMENT_VARIABLES[var_name]
            summary_lines.append(f"  {var_name}: {value} (default: {config['default']})")
        
        return "\\n".join(summary_lines)