#!/usr/bin/env python3
"""
Extract learned projection matrix from MambaASR checkpoint for CoreML deployment integration.

This script extracts the trained 1024→29 character projection weights from PyTorch checkpoints
and converts them to CSV format for integration with the Swift CoreML deployment pipeline.
It bridges the PyTorch training workflow with the production CoreML inference system.

Architecture integration:
- PyTorch training: MambaASR trains with learned projection head (1024→29 character mapping)
- Weight extraction: This script extracts the trained projection weights as log-softmax probabilities
- CoreML deployment: Swift MambaASRRunner loads CSV weights for character sequence decoding
- Production inference: CoreML model outputs 1024-class logits, CSV provides character mapping

Called by:
- Mamba-ASR-MPS/train.py post-training automation for projection extraction (line 559)
- Manual model export workflows when preparing trained models for deployment
- CI/CD pipelines for automated model training and export workflows
- Development workflows testing PyTorch→CoreML integration consistency

Calls to:
- PyTorch checkpoint loading for trained model state extraction
- torch.nn.functional.log_softmax for probability distribution computation
- CSV file writing for Swift/CoreML-compatible weight persistence

Mathematical transformation:
- Input: PyTorch Linear layer weights W[29, 1024] and bias b[29] (character vocab space)
- Processing: For each backbone token i, compute P[i,:] = log_softmax(W[:,i] + b) (character distribution)
- Output: CSV matrix P[1024, 29] where P[i,k] is log-probability of character k given backbone token i

Usage examples:
    # Standard extraction from trained checkpoint
    python Mamba-ASR-MPS/scripts/extract_projection_from_ckpt.py \
        --ckpt /path/to/best.pt \
        --w-key proj.weight \
        --b-key proj.bias \
        --out exports/projection_1024x29.csv

    # Extraction with weight matrix transpose (if needed)
    python Mamba-ASR-MPS/scripts/extract_projection_from_ckpt.py \
        --ckpt /path/to/model.pt \
        --w-key model.proj.weight \
        --transpose-w \
        --out deployment/projection.csv

Deployment integration:
- CSV output: 1024 rows × 29 columns, log-space probabilities for numerical stability
- Swift integration: MambaASRRunner.swift loads CSV for character sequence decoding
- CoreML compatibility: Log-space format prevents numerical underflow in mobile deployment
- Validation: Swift test harness compares PyTorch vs CoreML character sequence outputs

Technical constraints:
- Weight shape: Must be (29, 1024) after transpose handling (character_vocab × backbone_vocab)
- Bias shape: Optional (29,) for character-level bias terms, defaults to zeros if missing
- Numerical format: 8 decimal places for sufficient precision in mobile inference
- Memory efficiency: Processes weights in-place to minimize peak memory usage during extraction

Error handling:
- Checkpoint validation: Verifies file existence and loadable state dict format
- Key validation: Confirms weight/bias keys exist with helpful key listing on failure
- Shape validation: Enforces exact tensor shapes required for CoreML compatibility
- Graceful degradation: Missing bias treated as zeros rather than hard failure
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


class ProjectionExtractionConstants:
    """
    Named constants for projection matrix extraction and CoreML deployment integration.
    
    This class centralizes all projection extraction constants to eliminate magic numbers
    throughout the PyTorch→CoreML deployment pipeline. Constants are organized by category
    and provide clear documentation for vocabulary sizes, precision, and format requirements.
    
    Used throughout:
    - Weight and bias tensor shape validation during checkpoint loading
    - CSV output formatting and numerical precision configuration
    - CoreML deployment integration requiring specific matrix dimensions
    - Swift integration requiring consistent vocabulary size expectations
    
    Called by:
    - main() function for tensor shape validation and processing
    - CSV writing logic for numerical precision formatting
    - Error handling for shape mismatch reporting
    - CoreML deployment workflows requiring consistent matrix dimensions
    
    Architecture integration:
    - MambaASR training: Uses these vocabulary sizes for projection head architecture
    - PyTorch checkpoints: Trained weights conform to these dimensional constraints
    - CoreML deployment: Swift code expects matrices with these exact dimensions
    - Character tokenization: Character vocabulary size must match deployment expectations
    """
    
    # MARK: - Vocabulary Architecture Constants
    
    # Vocabulary size constants defining the projection matrix dimensions
    # These must match the MambaASR training configuration and CoreML deployment expectations
    BACKBONE_VOCAB_SIZE = 1024          # ConMamba CTC backbone output vocabulary size
    CHARACTER_VOCAB_SIZE = 29           # Character-level vocabulary size (a-z, space, apostrophe, blank)
    
    # Projection matrix shape expectations for validation
    EXPECTED_WEIGHT_SHAPE = (CHARACTER_VOCAB_SIZE, BACKBONE_VOCAB_SIZE)  # (29, 1024) after transpose handling
    EXPECTED_BIAS_SHAPE = (CHARACTER_VOCAB_SIZE,)                        # (29,) for character-level bias terms
    
    # MARK: - CSV Output Format Constants
    
    # Numerical precision constants for CoreML deployment compatibility
    CSV_DECIMAL_PRECISION = 8           # Decimal places for log-probability values (sufficient for mobile precision)
    LOG_SOFTMAX_DIMENSION = 0           # Dimension for log-softmax computation (over character vocabulary)
    
    # CSV matrix output dimensions for Swift integration
    CSV_OUTPUT_ROWS = BACKBONE_VOCAB_SIZE     # 1024 rows (one per backbone token)
    CSV_OUTPUT_COLS = CHARACTER_VOCAB_SIZE    # 29 columns (one per character)
    
    # MARK: - Default Value Constants
    
    # Default tensor values for missing or optional parameters
    DEFAULT_BIAS_DTYPE = torch.float32   # Data type for default bias when not provided in checkpoint
    CHECKPOINT_LOADING_DEVICE = "cpu"    # Device for checkpoint loading (CPU for memory efficiency)
    
    # MARK: - Error Handling Constants
    
    # String formatting constants for error messages and validation
    SHAPE_ERROR_FORMAT_PRECISION = 0    # No decimal places needed for tensor shape reporting
    CSV_LINE_SEPARATOR = "\n"           # Line separator for CSV file writing
    CSV_VALUE_SEPARATOR = ","           # Value separator for CSV file writing


def main() -> None:
    """
    Main function for extracting projection matrix from PyTorch checkpoint to CSV format.
    
    This function orchestrates the complete extraction pipeline from PyTorch checkpoint
    loading through CSV output generation for CoreML deployment integration.
    
    Processing workflow:
    1. Parse command line arguments for checkpoint path and tensor keys
    2. Load PyTorch checkpoint and validate state dict structure
    3. Extract weight and bias tensors with shape validation
    4. Compute log-softmax probability distributions for character mapping
    5. Write CSV output in format compatible with Swift CoreML integration
    
    Error handling:
    - Checkpoint file existence and loading validation
    - State dict key presence and tensor shape verification  
    - Graceful bias handling when optional bias key is missing
    - Helpful error messages with available keys when lookup fails
    """
    ap = argparse.ArgumentParser(
        description="Extract MambaASR projection matrix for CoreML deployment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Use named constants in help messages for consistency
    ap.add_argument("--ckpt", required=True, 
                   help="Path to PyTorch checkpoint (.pt/.pth)")
    ap.add_argument("--w-key", required=True, 
                   help=f"State dict key for projection weight "
                        f"({ProjectionExtractionConstants.CHARACTER_VOCAB_SIZE} x {ProjectionExtractionConstants.BACKBONE_VOCAB_SIZE} "
                        f"or {ProjectionExtractionConstants.BACKBONE_VOCAB_SIZE} x {ProjectionExtractionConstants.CHARACTER_VOCAB_SIZE} with --transpose-w)")
    ap.add_argument("--b-key", default=None, 
                   help=f"State dict key for projection bias ({ProjectionExtractionConstants.CHARACTER_VOCAB_SIZE},). Optional")
    ap.add_argument("--transpose-w", action="store_true", 
                   help=f"If set, transpose weight before extraction (from {ProjectionExtractionConstants.BACKBONE_VOCAB_SIZE}x{ProjectionExtractionConstants.CHARACTER_VOCAB_SIZE} "
                        f"to {ProjectionExtractionConstants.CHARACTER_VOCAB_SIZE}x{ProjectionExtractionConstants.BACKBONE_VOCAB_SIZE})")
    ap.add_argument("--out", required=True, 
                   help=f"Output CSV path ({ProjectionExtractionConstants.CSV_OUTPUT_ROWS} rows x {ProjectionExtractionConstants.CSV_OUTPUT_COLS} cols, log-space)")
    args = ap.parse_args()

    # Validate checkpoint file existence
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    # Load PyTorch checkpoint with flexible state dict handling
    checkpoint_obj = torch.load(str(ckpt_path), map_location=ProjectionExtractionConstants.CHECKPOINT_LOADING_DEVICE, weights_only=True)
    state_dict = checkpoint_obj.get("state_dict", checkpoint_obj)
    if not isinstance(state_dict, dict):
        raise SystemExit("Could not find state dict in checkpoint (expected dict or {'state_dict': dict})")

    # Extract and validate weight tensor
    if args.w_key not in state_dict:
        available_keys = "\n  - ".join(state_dict.keys())
        raise SystemExit(f"Weight key not found: {args.w_key}\nAvailable keys:\n  - {available_keys}")

    weight_tensor = state_dict[args.w_key].float()
    if args.transpose_w:
        weight_tensor = weight_tensor.t()
        
    if weight_tensor.shape != ProjectionExtractionConstants.EXPECTED_WEIGHT_SHAPE:
        raise SystemExit(
            f"Weight shape must be {ProjectionExtractionConstants.EXPECTED_WEIGHT_SHAPE} after transpose handling, "
            f"got {tuple(weight_tensor.shape)}"
        )

    # Extract and validate bias tensor (optional)
    if args.b_key is not None:
        if args.b_key not in state_dict:
            available_keys = "\n  - ".join(state_dict.keys())
            raise SystemExit(f"Bias key not found: {args.b_key}\nAvailable keys:\n  - {available_keys}")
            
        bias_tensor = state_dict[args.b_key].float()
        if bias_tensor.shape != ProjectionExtractionConstants.EXPECTED_BIAS_SHAPE:
            raise SystemExit(
                f"Bias shape must be {ProjectionExtractionConstants.EXPECTED_BIAS_SHAPE}, "
                f"got {tuple(bias_tensor.shape)}"
            )
    else:
        # Create default zero bias when not provided in checkpoint
        bias_tensor = torch.zeros(
            ProjectionExtractionConstants.CHARACTER_VOCAB_SIZE, 
            dtype=ProjectionExtractionConstants.DEFAULT_BIAS_DTYPE
        )

    # Compute log-softmax probability distributions for character mapping
    # For each backbone token i in 0..1023, compute P[i,:] = log_softmax(W[:,i] + b) over character vocabulary
    logits = weight_tensor + bias_tensor[:, None]  # Shape: (29, 1024) - broadcast bias across backbone vocab
    log_probabilities = F.log_softmax(logits, dim=ProjectionExtractionConstants.LOG_SOFTMAX_DIMENSION)  # Normalize over character vocab
    csv_matrix = log_probabilities.t().contiguous()  # Transpose to (1024, 29) for CSV output format

    # Write CSV output in format compatible with Swift CoreML integration
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w") as csv_file:
        for row_idx in range(csv_matrix.shape[0]):
            # Format each probability value with specified precision for mobile deployment
            formatted_values = [
                f"{float(value):.{ProjectionExtractionConstants.CSV_DECIMAL_PRECISION}f}" 
                for value in csv_matrix[row_idx].tolist()
            ]
            csv_row = ProjectionExtractionConstants.CSV_VALUE_SEPARATOR.join(formatted_values)
            csv_file.write(csv_row + ProjectionExtractionConstants.CSV_LINE_SEPARATOR)
            
    print(f"Wrote projection CSV: {output_path}")
    print(f"Matrix dimensions: {csv_matrix.shape[0]} rows × {csv_matrix.shape[1]} columns")
    print(f"Precision: {ProjectionExtractionConstants.CSV_DECIMAL_PRECISION} decimal places")


if __name__ == "__main__":
    main()
