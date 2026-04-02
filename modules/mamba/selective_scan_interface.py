"""
Pure-PyTorch reference implementation of Mamba's selective scan.

This module provides the core selective scan operation that defines Mamba's
sequence modeling capabilities. This is Phase 1 of the Apple Silicon implementation -
a functional baseline that trades performance for correctness and device compatibility.

Architectural Context:
- Called by: MambaBlock.forward() in mamba_blocks.py:67
- Used in: ConMambaCTC and MCTModel architectures 
- Dependencies: Pure PyTorch tensors, no CUDA-specific operations
- Device Support: CPU, CUDA, MPS (Apple Silicon)

Implementation Philosophy:
- This is a "naive" implementation using sequential Python loops
- Designed for functional validation, not performance
- Follows the mathematical formulation from the Mamba paper
- Intentionally avoids optimization to maintain clarity for AI developers

Performance Expectations:
- Sequential loop creates dispatch overhead on GPU
- Memory bandwidth bound due to lack of kernel fusion
- Expected 10-100x slower than optimized Metal kernels
- Suitable for prototyping, not production training

Future Optimization Path:
- Phase 2: Custom Metal kernel with parallel scan algorithm
- Phase 3: Kernel fusion for memory bandwidth optimization
- See README/Mamba-on-Apple-Silicon.md Section 3.2 for Metal implementation

References:
- Mathematical foundation: Mamba paper selective scan formulation
- Hardware optimization: README/Mamba-on-Apple-Silicon.md Section 1.2
- Apple Silicon guide: README/Mamba-on-Apple-Silicon.md Section 3.1
"""
from __future__ import annotations

from typing import Tuple

import torch
import os
from contextlib import nullcontext
if os.environ.get("MAMBA_DISABLE_RECORD_FUNCTION", "0") == "1":
    record_function = nullcontext  # type: ignore
else:
    from torch.autograd.profiler import record_function  # type: ignore
import torch.nn.functional as F


# Mathematical and Implementation Constants
class SelectiveScanConstants:
    """Named constants for selective scan implementation.
    
    These values are derived from the Mamba paper and empirical Apple Silicon testing.
    All constants include explanations to guide future AI developers.
    """
    
    # Numerical Stability
    SOFTPLUS_THRESHOLD = 20.0  # Above this value, softplus(x) ≈ x (numerical stability)
    MIN_DELTA = 1e-8  # Minimum discretization step to prevent numerical underflow
    
    # Tensor Shape Documentation
    # These are not used in computation but document expected tensor dimensions
    BATCH_DIM = 0      # Batch dimension index in tensors
    SEQ_DIM = 1        # Sequence length dimension index  
    MODEL_DIM = 2      # Model/feature dimension index
    STATE_DIM = 3      # State space dimension index (for 4D tensors)
    
    # Memory Layout Constants
    # Apple Silicon unified memory considerations
    EINSUM_THRESHOLD = 1000  # Below this size, use explicit broadcasting over einsum
    
    @staticmethod
    def get_numerical_info() -> str:
        """Return documentation about numerical choices for AI developers."""
        return """
        Numerical Implementation Notes:
        - SOFTPLUS_THRESHOLD: Based on float32 precision limits
        - MIN_DELTA: Prevents state explosion in long sequences
        - Einsum vs Broadcasting: Apple Silicon memory bandwidth optimization
        """


def selective_scan(
    x: torch.Tensor,        # (B, L, D) - Input sequence features
    delta: torch.Tensor,    # (B, L, D) - Time-varying discretization steps  
    A: torch.Tensor,        # (D, N) - State transition matrix (time-invariant)
    B_proj: torch.Tensor,   # (B, L, N) - Input-to-state projection (time-varying)
    C_proj: torch.Tensor,   # (B, L, N) - State-to-output projection (time-varying) 
    D: torch.Tensor,        # (D,) - Skip connection weights
    z: torch.Tensor,        # (B, L, D) - Gating/selection mechanism
    delta_bias: torch.Tensor,  # (D,) - Bias for discretization stability
    h0: torch.Tensor,       # (B, D, N) - Initial hidden state
) -> torch.Tensor:
    """Core selective scan operation implementing Mamba's sequence modeling.
    
    This function implements the mathematical heart of the Mamba architecture:
    the selective state space model that enables content-aware sequence processing.
    
    Mathematical Formulation:
    1. Discretization: Δ = softplus(δ + bias), A_discrete = exp(Δ * A)
    2. State Update: h_t = A_discrete_t * h_{t-1} + Δ_t * B_t * x_t  
    3. Output: y_t = C_t^T * h_t + D * x_t
    4. Gating: output = y * sigmoid(z)
    
    Implementation Strategy:
    - Sequential loop over time dimension (not parallel)
    - Uses explicit tensor operations for MPS compatibility
    - Avoids complex einsum operations that may fall back to CPU
    - Prioritizes numerical stability over raw performance
    
    Performance Characteristics:
    - Time Complexity: O(B * L * D * N) - sequential in L
    - Memory Complexity: O(B * D * N) - state size independent of sequence length
    - GPU Utilization: Low due to Python loop dispatch overhead
    - MPS Fallback Risk: Minimal - uses only well-supported operations
    
    Apple Silicon Considerations:
    - Unified memory reduces CPU-GPU transfer cost
    - Small tensor operations may be slower than CPU
    - Kernel dispatch overhead dominates computation time
    - Suitable for validation, not production performance
    
    Called By:
    - MambaBlock.forward() at mamba_blocks.py:67
    - Used in both CTC and RNN-T training pipelines
    
    Next Phase:
    - Replace with fused Metal kernel for 10-100x speedup
    - Implement parallel scan algorithm for true GPU utilization
    - See README/Mamba-on-Apple-Silicon.md Section 3.2
    
    Args:
        x: Input sequence features (B=batch, L=seq_len, D=model_dim)
        delta: Discretization steps, controls state update rate per timestep
        A: Time-invariant state transition matrix (shared across sequence)
        B_proj: Time-varying input projection (content-dependent)
        C_proj: Time-varying output projection (content-dependent)
        D: Skip connection weights for residual path
        z: Gating values for selective information flow
        delta_bias: Learned bias for discretization numerical stability
        h0: Initial hidden state (typically zeros)
        
    Returns:
        y: Output sequence (B, L, D) with selective state space processing
        
    Tensor Shape Evolution:
        Input:  x(B,L,D) -> delta(B,L,D) -> A(D,N) -> B_proj(B,L,N) -> C_proj(B,L,N)
        State:  h0(B,D,N) -> h_t(B,D,N) [updated each timestep]
        Output: y_t(B,D) -> y(B,L,D) [stacked over sequence]
    """
    # Extract tensor dimensions for clarity and validation
    batch_size, seq_len, d_model = x.shape
    state_dim = A.shape[1]
    
    # Validate tensor shapes for debugging and AI developer clarity
    assert delta.shape == (batch_size, seq_len, d_model), f"delta shape mismatch: {delta.shape}"
    assert A.shape == (d_model, state_dim), f"A shape mismatch: {A.shape}"
    assert B_proj.shape == (batch_size, seq_len, state_dim), f"B_proj shape mismatch: {B_proj.shape}"
    assert C_proj.shape == (batch_size, seq_len, state_dim), f"C_proj shape mismatch: {C_proj.shape}"
    assert D.shape == (d_model,), f"D shape mismatch: {D.shape}"
    assert z.shape == (batch_size, seq_len, d_model), f"z shape mismatch: {z.shape}"
    assert delta_bias.shape == (d_model,), f"delta_bias shape mismatch: {delta_bias.shape}"
    assert h0.shape == (batch_size, d_model, state_dim), f"h0 shape mismatch: {h0.shape}"

    # Profiling annotation for performance analysis on Apple Silicon
    # Enables detailed timing analysis via PyTorch profiler and Instruments
    with record_function("selective_scan_naive"):
        # Step 1: Discretize parameters using softplus for numerical stability
        with record_function("ss_softplus_discretize"):
            delta_positive = F.softplus(delta + delta_bias.view(1, 1, -1))  # (B, L, D)
            delta_positive = torch.clamp(delta_positive, min=SelectiveScanConstants.MIN_DELTA)

        # Step 2: Compute discretized state transition matrices
        with record_function("ss_state_transition_exp"):
            delta_A = (delta_positive.unsqueeze(-1) * A.view(1, 1, A.shape[0], A.shape[1])).exp()
            delta_A = torch.clamp(delta_A, max=1e10)

        # Step 3: Compute discretized input projection
        # delta_B_u must be (B, L, D, N) to preserve per-channel selectivity
        with record_function("ss_input_proj"):
            delta_u = delta_positive * x  # (B, L, D)
            delta_B_u = delta_u.unsqueeze(-1) * B_proj.unsqueeze(2)  # (B, L, D, N)

        # Step 4: Initialize state and output accumulation
        hidden_state = h0.clone()  # (B, D, N)
        output_timesteps = []

        # Step 5: Sequential state update loop (bottleneck)
        with record_function("ss_time_loop"):
            for timestep in range(seq_len):
                hidden_state = delta_A[:, timestep] * hidden_state + delta_B_u[:, timestep]
                C_timestep = C_proj[:, timestep, :]  # (B, N)
                # Implementation switch for inner product: default to einsum.
                # Set env MAMBA_EINSUM_IMPL=bmm to force batched matmul path.
                if os.environ.get("MAMBA_EINSUM_IMPL", "einsum").lower() == "bmm":
                    # Shapes: hidden_state (B, D, N) @ C_timestep.unsqueeze(-1) (B, N, 1) -> (B, D, 1)
                    y_timestep = torch.bmm(hidden_state, C_timestep.unsqueeze(-1)).squeeze(-1)
                else:
                    y_timestep = torch.einsum("bdn,bn->bd", hidden_state, C_timestep)
                output_timesteps.append(y_timestep)

        # Step 6: Combine outputs and apply residual connections
        with record_function("ss_output_post"):
            output_sequence = torch.stack(output_timesteps, dim=1)  # (B, L, D)
            skip_connection = x * D.view(1, 1, -1)
            output_with_skip = output_sequence + skip_connection
            gating_weights = torch.sigmoid(z)
            final_output = output_with_skip * gating_weights
        return final_output  # (B, L, D)


def init_hidden(batch_size: int, d_model: int, state_dim: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Initialize hidden state for selective scan operation.
    
    Creates the initial state tensor h0 that serves as the starting point
    for the recurrent state updates in the selective scan.
    
    Design Decisions:
    - Zero initialization is standard for SSMs and provides stable training
    - Alternative initializations (small random, learned) could be explored
    - Device placement is explicit to ensure MPS compatibility
    
    Called By:
    - MambaBlock.forward() at mamba_blocks.py:55
    - Called once per forward pass, before selective_scan()
    
    Memory Considerations:
    - State tensor is (B, D, N) which can be large for long sequences
    - Apple Silicon unified memory enables larger states than discrete GPU
    - Memory usage scales with batch size and model dimensions
    
    Args:
        batch_size: Number of sequences in batch
        d_model: Model dimension (feature size)
        state_dim: State space dimension (typically 16-64)
        device: Target device (cpu, cuda, or mps)
        
    Returns:
        h0: Initial hidden state tensor (B, D, N) filled with zeros
        
    Example Usage:
        h0 = init_hidden(32, 256, 16, torch.device('mps'))
        # Creates (32, 256, 16) zero tensor on Apple Silicon GPU
    """
    # Use zeros initialization - standard practice for state space models
    # This provides stable gradients and avoids initial activation explosions
    return torch.zeros(batch_size, d_model, state_dim, device=device, dtype=dtype)


def get_selective_scan_info() -> str:
    """Return implementation documentation for AI developers.
    
    This function provides runtime documentation about the selective scan
    implementation choices, performance characteristics, and optimization roadmap.
    
    Returns:
        Comprehensive documentation string for AI developers
    """
    return f"""
    Selective Scan Implementation Summary:
    
    Current Status: Phase 1 Functional Baseline
    - Pure PyTorch implementation for device compatibility
    - Sequential loop creates GPU dispatch overhead
    - Suitable for prototyping and numerical validation
    - Expected 10-100x slower than optimized implementation
    
    Mathematical Foundation:
    - Implements Mamba paper selective scan formulation
    - Time-varying parameters enable content-aware processing
    - Linear time complexity O(L) in theory, O(L) dispatch overhead in practice
    
    Apple Silicon Optimizations:
    - Device-agnostic code works on MPS, CUDA, CPU
    - Explicit broadcasting avoids einsum MPS fallbacks
    - Numerical stability for float32 precision
    - Constants: {SelectiveScanConstants.get_numerical_info()}
    
    Performance Profile:
    - Memory bandwidth bound due to lack of kernel fusion
    - Small tensor dispatch overhead dominates on GPU
    - Unified memory architecture reduces transfer costs
    
    Next Phase Optimizations:
    - Custom Metal kernel with parallel scan algorithm
    - Kernel fusion for 10-50x memory bandwidth improvement
    - See README/Mamba-on-Apple-Silicon.md Section 3.2
    
    Integration Points:
    - Called by MambaBlock in core training loop
    - Used in both CTC and RNN-T model architectures
    - Critical path for training performance
    """
