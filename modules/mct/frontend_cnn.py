"""
CNN frontend for acoustic feature processing in speech recognition on Apple Silicon.

This module implements a convolutional neural network frontend optimized for
speech recognition preprocessing. It performs feature extraction and time
subsampling to prepare acoustic features for subsequent Mamba encoder processing.

Architectural Role:
- Called by: MCTModel.forward() in mct_model.py for acoustic preprocessing
- Used in: RNN-T speech recognition pipeline as the first processing stage
- Integration: Feeds into MambaEncoder for sequence modeling
- Apple Neural Engine: Designed for ANE compatibility where applicable

Time Subsampling Strategy:
- Input: (B, T, 80) raw mel-spectrogram features
- Stage 1: Conv1d with stride=2, reduces T -> T/2
- Stage 2: Conv1d with stride=2, reduces T/2 -> T/4
- Stage 3: Conv1d with stride=1, feature refinement
- Output: (B, T/4, D) processed features for Mamba encoder

Apple Silicon Optimizations:
- Conv1d operations leverage Metal Performance Shaders
- GELU activation has efficient MPS implementation
- Layer normalization optimized for unified memory architecture
- Kernel sizes chosen for optimal MPS dispatch efficiency

Performance Characteristics:
- Computational complexity: O(B * T * D) - linear in sequence length
- Memory reduction: 4x sequence length compression
- GPU utilization: Well-optimized convolution operations
- Bottleneck analysis: Minimal compared to subsequent Mamba processing

Design Rationale:
- 4x subsampling reduces Mamba computational load significantly
- GELU activation provides better gradients than ReLU
- Multiple Conv1d stages for progressive feature abstraction
- LayerNorm stabilizes training dynamics

Called By:
- MCTModel.forward() at mct_model.py for RNN-T acoustic modeling
- Training pipelines via MCTModel instantiation
- Inference engines for speech recognition preprocessing

References:
- Convolution optimization: Metal Performance Shaders documentation
- Speech preprocessing: Standard mel-spectrogram to feature pipeline
- Apple Silicon guide: README/Mamba-on-Apple-Silicon.md Section 2
"""
from __future__ import annotations

import torch
import torch.nn as nn


# Frontend Configuration Constants
class FrontendConstants:
    """Named constants for CNN frontend architecture and optimization.
    
    These constants define the convolutional architecture parameters
    optimized for speech recognition on Apple Silicon hardware.
    """
    
    # Input/Output Dimensions
    DEFAULT_MEL_FEATURES = 80       # Standard mel-spectrogram feature count
    DEFAULT_MODEL_DIM = 256         # Output model dimension
    
    # Convolution Parameters
    STAGE1_KERNEL = 5               # First conv layer kernel size
    STAGE2_KERNEL = 5               # Second conv layer kernel size  
    STAGE3_KERNEL = 3               # Third conv layer kernel size
    
    SUBSAMPLING_STRIDE = 2          # Stride for time subsampling layers
    REFINEMENT_STRIDE = 1           # Stride for feature refinement layer
    
    # Padding Configuration
    STAGE1_PADDING = 2              # Padding for kernel size 5
    STAGE2_PADDING = 2              # Padding for kernel size 5
    STAGE3_PADDING = 1              # Padding for kernel size 3
    
    # Architecture Scaling
    TOTAL_SUBSAMPLING_FACTOR = 4    # Total time reduction: 2 * 2 = 4x
    NUM_CONV_STAGES = 3             # Number of convolutional stages
    
    @staticmethod
    def get_subsampling_info() -> str:
        """Return time subsampling documentation."""
        return f"""
        CNN Frontend Time Subsampling:
        - Stage 1: T -> T/2 (stride {FrontendConstants.SUBSAMPLING_STRIDE}, kernel {FrontendConstants.STAGE1_KERNEL})
        - Stage 2: T/2 -> T/4 (stride {FrontendConstants.SUBSAMPLING_STRIDE}, kernel {FrontendConstants.STAGE2_KERNEL})
        - Stage 3: T/4 -> T/4 (stride {FrontendConstants.REFINEMENT_STRIDE}, kernel {FrontendConstants.STAGE3_KERNEL})
        - Total reduction: {FrontendConstants.TOTAL_SUBSAMPLING_FACTOR}x
        - Benefit: Reduces Mamba sequence length for efficiency
        """


class FrontendCNN(nn.Module):
    """CNN frontend for acoustic feature processing with Apple Silicon optimization.
    
    This module implements a three-stage convolutional frontend that transforms
    raw mel-spectrogram features into processed representations suitable for
    Mamba sequence modeling.
    
    Architecture Design:
    - Stage 1: Feature expansion and 2x time subsampling
    - Stage 2: Feature refinement and additional 2x time subsampling
    - Stage 3: Feature consolidation with no time reduction
    - Output: Layer normalization for training stability
    
    Apple Silicon Optimizations:
    - Conv1d operations leverage Metal Performance Shaders
    - GELU activation functions have efficient MPS implementations
    - Kernel sizes optimized for MPS memory access patterns
    - Layer normalization benefits from Accelerate framework
    
    Performance Profile:
    - Time complexity: O(B * T * D) where D is model dimension
    - Memory usage: O(B * T * D) peak during intermediate stages
    - GPU utilization: High for convolution operations
    - Bottleneck: Minimal compared to subsequent Mamba processing
    
    Integration Context:
    - Input: Raw mel-spectrogram features (B, T, 80)
    - Output: Processed features for MambaEncoder (B, T/4, D)
    - Called by: MCTModel.forward() in the RNN-T pipeline
    - Prepares data for: MambaEncoder.forward() processing
    
    Memory Efficiency:
    - 4x sequence length reduction crucial for Mamba linear complexity
    - Reduces memory bandwidth requirements for subsequent processing
    - Enables longer audio sequences on unified memory architecture
    """
    
    def __init__(self, feat_dim: int = FrontendConstants.DEFAULT_MEL_FEATURES, 
                 d_model: int = FrontendConstants.DEFAULT_MODEL_DIM):
        """Initialize CNN frontend with configurable dimensions.
        
        Args:
            feat_dim: Input feature dimension (typically 80 mel features)
            d_model: Output model dimension for subsequent processing
        """
        super().__init__()
        
        # Store configuration for reference
        self.feat_dim = feat_dim
        self.d_model = d_model
        
        # Three-stage convolutional processing network
        # Each stage: Conv1d -> GELU activation
        self.conv_network = nn.Sequential(
            # Stage 1: Feature expansion with 2x time subsampling
            nn.Conv1d(
                feat_dim, 
                d_model, 
                kernel_size=FrontendConstants.STAGE1_KERNEL,
                stride=FrontendConstants.SUBSAMPLING_STRIDE,
                padding=FrontendConstants.STAGE1_PADDING
            ),
            nn.GELU(),  # Smooth activation, well-optimized on Apple Silicon
            
            # Stage 2: Feature refinement with additional 2x time subsampling
            nn.Conv1d(
                d_model, 
                d_model,
                kernel_size=FrontendConstants.STAGE2_KERNEL,
                stride=FrontendConstants.SUBSAMPLING_STRIDE,
                padding=FrontendConstants.STAGE2_PADDING
            ),
            nn.GELU(),
            
            # Stage 3: Feature consolidation with no time reduction
            nn.Conv1d(
                d_model, 
                d_model,
                kernel_size=FrontendConstants.STAGE3_KERNEL,
                stride=FrontendConstants.REFINEMENT_STRIDE,
                padding=FrontendConstants.STAGE3_PADDING
            ),
            nn.GELU(),
        )
        
        # Output normalization for training stability
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Process mel-spectrogram features through CNN frontend.
        
        Transforms input mel-spectrograms through three-stage convolution
        processing with time subsampling for efficient Mamba processing.
        
        Args:
            feats: Input mel-spectrogram features (B, T, feat_dim)
                   B=batch_size, T=time_frames, feat_dim=80 (mel features)
        
        Returns:
            Processed features (B, T/4, d_model) ready for Mamba encoder
            
        Tensor Flow:
            Input: feats(B,T,F) -> transpose -> (B,F,T)
            Conv: (B,F,T) -> conv_network -> (B,D,T/4)
            Output: (B,D,T/4) -> transpose -> (B,T/4,D) -> norm -> (B,T/4,D)
        
        Performance Notes:
        - Conv1d expects (batch, channels, time) format
        - Transpose operations are lightweight on Apple Silicon
        - Time subsampling reduces sequence length by 4x
        - LayerNorm provides training stability
        """
        # Validate input dimensions
        batch_size, time_frames, feature_dim = feats.shape
        assert feature_dim == self.feat_dim, f"Feature dim mismatch: {feature_dim} != {self.feat_dim}"
        
        # Step 1: Transpose for Conv1d processing
        # Conv1d expects (batch, channels, sequence) but we have (batch, sequence, features)
        conv_input = feats.transpose(1, 2)  # (B, feat_dim, T)
        
        # Step 2: Apply three-stage convolutional processing
        # This performs feature extraction and 4x time subsampling
        conv_output = self.conv_network(conv_input)  # (B, d_model, T/4)
        
        # Step 3: Transpose back to sequence-first format
        # Mamba encoder expects (batch, sequence, features)
        sequence_output = conv_output.transpose(1, 2)  # (B, T/4, d_model)
        
        # Step 4: Apply output normalization
        # LayerNorm stabilizes training and improves convergence
        normalized_output = self.output_norm(sequence_output)  # (B, T/4, d_model)
        
        return normalized_output
    
    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length after frontend processing.

        Uses the actual Conv1d output formula: floor((L + 2*pad - kernel) / stride) + 1

        Args:
            input_length: Input sequence length

        Returns:
            Output sequence length after frontend convolutions
        """
        # Stage 1: Conv1d(kernel=5, stride=2, pad=2)
        length = (input_length + 2 * FrontendConstants.STAGE1_PADDING - FrontendConstants.STAGE1_KERNEL) // FrontendConstants.SUBSAMPLING_STRIDE + 1
        # Stage 2: Conv1d(kernel=5, stride=2, pad=2)
        length = (length + 2 * FrontendConstants.STAGE2_PADDING - FrontendConstants.STAGE2_KERNEL) // FrontendConstants.SUBSAMPLING_STRIDE + 1
        # Stage 3: Conv1d(kernel=3, stride=1, pad=1) — preserves length
        length = (length + 2 * FrontendConstants.STAGE3_PADDING - FrontendConstants.STAGE3_KERNEL) // FrontendConstants.REFINEMENT_STRIDE + 1
        return length
    
    def get_frontend_info(self) -> str:
        """Return frontend configuration and performance information."""
        total_params = sum(p.numel() for p in self.parameters())
        return f"""
        CNN Frontend Configuration:
        - Input dimension: {self.feat_dim} (mel features)
        - Output dimension: {self.d_model} (model features)
        - Subsampling factor: {FrontendConstants.TOTAL_SUBSAMPLING_FACTOR}x
        - Convolution stages: {FrontendConstants.NUM_CONV_STAGES}
        - Total parameters: {total_params:,}
        - Memory efficiency: Reduces sequence length for Mamba processing
        - Apple Silicon: Optimized for MPS backend
        {FrontendConstants.get_subsampling_info()}
        """
