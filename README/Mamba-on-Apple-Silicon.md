# An AI Research Engineer's Field Guide: High-Performance Mamba on Apple Silicon

This guide addresses a central engineering challenge in modern sequence modeling: adapting architectures whose performance is fundamentally tied to specific hardware ecosystems. The Mamba architecture's state-of-the-art capabilities are not merely algorithmic; they are a product of a hardware-aware implementation of its core selective scan algorithm, meticulously optimized for NVIDIA's CUDA platform.1 A direct, naive port to any other platform, including Apple Silicon, is destined for failure. This document serves as a comprehensive engineering roadmap for successfully porting, implementing, and optimizing Mamba on Apple's M-series Systems-on-a-Chip (SoCs). We will dissect the core algorithm, navigate the Apple ML ecosystem, provide two distinct, high-performance implementation paths, and arm the developer with the tools to benchmark, debug, and validate their work. This guide transforms a formidable research engineering task into a well-defined and achievable project.

## Section 1: Deconstructing the Core Challenge: The Selective Scan Algorithm

To successfully implement Mamba on a new hardware platform, it is imperative to first understand that its performance is not an emergent property of its mathematical formulation alone. It is the result of a sophisticated co-design of its core algorithm with the underlying hardware architecture. This section dissects the Mamba block to isolate its performance-critical component—the selective scan—and analyzes why the reference implementation is both a marvel of efficiency and a significant porting barrier. A deep understanding of this foundational problem is the prerequisite for any successful implementation on non-CUDA hardware.

### 1.1 The Anatomy of a Mamba Block: Beyond a Simple Recurrence

The Mamba architecture represents a significant step in the evolution of State Space Models (SSMs) for sequence modeling.3 At its core, an SSM maps a 1D input sequence,

u(t), to a 1D output sequence, y(t), through an N-dimensional latent state, h(t). The dynamics of this system are governed by a set of linear ordinary differential equations (ODEs) in the continuous-time domain 4:

h′(t)=Ah(t)+Bu(t)

y(t)=Ch(t)+Du(t)

Here, A, B, C, and D are matrices that define the system's behavior. For practical application in deep learning, this continuous-time representation must be discretized. Mamba, like its predecessors, employs a discretization method (specifically, the zero-order hold) to transform these ODEs into a recurrent neural network (RNN) formulation.5 This results in the familiar discrete-time update rule:

ht=Aˉht−1+Bˉxt

yt=Cht+Dxt

where Aˉ and Bˉ are the discretized versions of the continuous-time matrices A and B, and xt represents the input token at timestep t.

The revolutionary aspect of Mamba lies in its "selection" mechanism.2 In prior SSMs like S4, the state matrices (

A,B,C) were time-invariant, meaning they were fixed parameters for all inputs in a sequence. This property allowed the recurrent computation to be unrolled into a large, efficient convolution, which is highly parallelizable on GPUs.9 Mamba breaks this time-invariance by making the key parameters—specifically the discretization step size

Δ, the input projection B, and the output projection C—functions of the input token xt itself.4 This input-dependency allows the model to dynamically "select" which information to keep in its state and which to forget, giving it the ability to perform content-aware reasoning similar to the attention mechanism in Transformers, but with a theoretical linear-time complexity

O(L) in sequence length L instead of the quadratic O(L2).10

However, this powerful selection mechanism comes at a steep implementation cost. By making the system parameters time-varying, Mamba forfeits the ability to be represented as a global convolution.9 It is forced back into a recurrent mode of computation, which is inherently sequential and notoriously slow to train on parallel hardware. The entire performance of the Mamba architecture hinges on overcoming this limitation.

This is where the `selective_scan` operation comes into play. Architecturally, a Mamba block is a simple and homogeneous structure that combines the SSM for sequence mixing with a gated MLP for feature mixing, often using a SiLU/Swish activation function.2 Within this block, the

`selective_scan` is the specific, highly-optimized algorithm designed to execute the input-dependent recurrence efficiently. It is this single component that is responsible for both Mamba's impressive performance and its profound implementation difficulty on new platforms.

### 1.2 The "Hardware-Aware" Imperative: A Deep Dive into the CUDA Reference Implementation

The term "hardware-aware" is not a mere buzzword in the context of Mamba; it is the central pillar of its practical success. The performance gains are not solely from reducing the theoretical FLOP count but, more importantly, from a deep understanding and optimization of the data movement within the GPU memory hierarchy.1 This philosophy is analogous to that of FlashAttention, where the primary bottleneck in modern deep learning is often not computation but memory I/O—the time spent moving data between the large but slow High-Bandwidth Memory (HBM) and the small but extremely fast on-chip SRAM.15 The official Mamba implementation leverages three classical techniques to master this data movement.

### Pillar 1: Kernel Fusion

The most significant bottleneck for a naive implementation of the selective scan is memory bandwidth. The intermediate state tensor, h, has a shape of (Batch Size, Sequence Length, Model Dimension, State Dimension), or `(B, L, D, N)`. Since the state dimension `N` is typically 16 or larger, this tensor is significantly larger than the input or output. A standard implementation would involve multiple separate GPU operations: one to compute the discretized Aˉ and Bˉ, another to run the scan operation, and a third to multiply the resulting state by C. Each of these steps would require reading inputs from HBM and writing the full intermediate state back to HBM, resulting in memory traffic on the order of O(BLDN) bytes per step.

The Mamba reference implementation avoids this by fusing these disparate operations into a single, monolithic CUDA kernel.1 This fused kernel executes the following sequence:

1. **Load:** The SSM parameters (Δ,A,B,C) are loaded a single time from the slow HBM into the fast on-chip SRAM.
2. **Compute in SRAM:** The discretization of A and B, the scan recurrence itself, and the final multiplication with C are all performed entirely within SRAM, without writing the large intermediate state h back to HBM.
3. **Write:** Only the final output y (of shape `B, L, D`) is written back to HBM.

By keeping the massive intermediate state localized to the fast on-chip memory, kernel fusion reduces the total memory I/O by a factor proportional to the state dimension N. This single optimization is responsible for a reported 20-40x speedup over a standard, non-fused implementation and is the primary source of Mamba's efficiency.2

### Pillar 2: Parallel Scan

The second challenge is the sequential nature of the recurrence, where calculating ht requires the value of ht−1. This dependency seems to preclude parallelization across the time dimension. The solution is to re-frame the scan operation as an associative operator, which can then be solved efficiently in parallel using a prefix sum (or scan) algorithm.5

The Mamba implementation uses a work-efficient parallel scan algorithm, such as the one proposed by Blelloch.9 This algorithm works in two phases: an "up-sweep" or reduction phase, where partial sums are computed in a tree-like fashion, and a "down-sweep" phase, where the final prefix sums are constructed from the intermediate values. This transforms a computation with

O(L) sequential steps into a parallel computation with a depth of O(logL), making it perfectly suited for the massively parallel architecture of a GPU.18 This is the key algorithmic trick that allows Mamba to be trained in parallel like a Transformer, a feat not possible with traditional RNNs like LSTMs.18

### Pillar 3: Activation Recomputation

The final piece of the puzzle is managing memory during the backward pass (backpropagation). To compute gradients, the backward pass requires access to the intermediate hidden states h that were calculated during the forward pass. Storing all of these states would result in a memory footprint of O(BLDN), which would be enormous for long sequences and would negate Mamba's memory efficiency.

To solve this, the hardware-aware algorithm employs activation recomputation. During the forward pass, the intermediate states h are computed in SRAM but are *not* saved to HBM.1 During the backward pass, when the gradients are needed, the forward computation for a given segment of the sequence is run again, recomputing the necessary intermediate states on-the-fly within SRAM. This approach trades a small amount of extra computation (which is fast, as it's mostly happening in SRAM) for a massive reduction in memory I/O and overall memory footprint. This technique is what allows a Mamba block to have a memory footprint comparable to that of a highly optimized Transformer block using FlashAttention, making it viable for training on very long sequences.2

### 1.3 The Porting Bottleneck: Why a Naive Implementation Will Fail

The synthesis of these three pillars—kernel fusion, parallel scan, and activation recomputation—results in a highly specialized, high-performance CUDA kernel. The `selective_scan_interface.py` file in the official Mamba repository is a thin Python wrapper that calls this compiled CUDA code.14 This presents the central challenge for porting Mamba to Apple Silicon: the core of the model's performance is locked away in non-portable, NVIDIA-specific code.

An attempt to create a "naive" implementation on Apple Silicon using a high-level framework like PyTorch with its Metal Performance Shaders (MPS) backend will inevitably fail to achieve the required performance for several reasons:

1. **Lack of Fusion:** A standard implementation using distinct PyTorch operations (`torch.einsum`, `torch.cumsum`, etc.) will result in multiple separate Metal kernel dispatches. The framework will not automatically fuse these operations. This will force the materialization of the large intermediate state h in Apple's Unified Memory (the equivalent of HBM), creating the exact memory I/O bottleneck the original implementation was designed to avoid.
2. **Inefficient Scan:** While PyTorch MPS may have an implementation for `cumsum`, it is unlikely to be optimized for the specific associative operator required by Mamba's scan. More likely, a developer would resort to a Python `for` loop, which would serialize the computation and introduce catastrophic overhead from dispatching thousands of tiny kernels.
3. **Memory Explosion:** Without a custom backward pass that implements recomputation, the standard autograd engine will attempt to save the intermediate state h for the backward pass, leading to an explosion in memory usage and likely causing out-of-memory errors even for moderately long sequences.

This leads to a critical conclusion for the developer: high-performance Mamba on Apple Silicon is *contingent* on creating a custom, fused Metal kernel that replicates the *principles* of the CUDA implementation. It is not an optional optimization; it is a fundamental prerequisite for success. The task is not one of simple algorithmic translation but of deep architectural re-engineering for a new hardware platform.

The performance gap between a naive implementation and a hardware-aware one is not incremental; it is a chasm. The research clearly separates Mamba's algorithmic design from its practical performance, which is credited entirely to this hardware-aware implementation.1 Simultaneously, user reports and documentation for Apple's ML frameworks, particularly PyTorch/MPS, highlight a history of performance inconsistencies and missing operator implementations.20 A direct causal link can be drawn: applying a naive software implementation to an algorithm that explicitly requires hardware-specific optimization will result in a performance profile that is not just suboptimal, but likely

*worse* than less advanced architectures. The developer will be paying the complexity cost of Mamba without reaping any of its speed benefits. Therefore, the project's first milestone cannot be simply "get Mamba running." It must be "get a baseline Mamba running *and profile it* to quantitatively prove the selective scan is the bottleneck." This data-driven step is essential to justify the significant engineering effort of writing a custom kernel.

## Section 2: Navigating the Apple Silicon ML Ecosystem

Before embarking on the implementation, a developer must possess a clear mental model of the target platform. Apple Silicon is not merely a different brand of CPU or GPU; it is a fundamentally different System-on-a-Chip (SoC) architecture with unique strengths, weaknesses, and a distinct software ecosystem. This section provides a concise, ML-centric overview of the M-series hardware and the two viable software stacks, highlighting the strategic decisions, opportunities, and pitfalls that will define the project's success.

### 2.1 An ML Engineer's Guide to the M-Series Architecture

Apple's M-series chips integrate the CPU, GPU, Neural Engine, and memory onto a single piece of silicon.23 This tight integration enables architectural features that differ significantly from the traditional discrete CPU/GPU model found in most ML workstations.

### Unified Memory Architecture (UMA): The Double-Edged Sword

The most significant architectural feature of Apple Silicon for deep learning is its Unified Memory Architecture.24 In a traditional PC, the CPU has its own RAM, and a discrete GPU has its own dedicated Video RAM (VRAM). Moving data between them requires an explicit copy over a PCIe bus, which can be a significant bottleneck. In UMA, a single pool of high-bandwidth memory is accessible by the CPU, GPU, and other processors on the chip.27

This design presents a critical trade-off:

- **Pro: Massive Memory Capacity.** The most immediate benefit is the sheer amount of memory available to the GPU. A MacBook Pro or Mac Studio can be configured with up to 128GB or 192GB of unified memory, respectively.26 This entire pool is accessible to the GPU, dwarfing the 24GB or 48GB of VRAM found on even high-end consumer and professional NVIDIA cards. This is a game-changer for working with large models or, in Mamba's case, extremely long sequences whose state might not fit in traditional VRAM.29
- **Pro: Zero-Copy Data Sharing.** Because the CPU and GPU share the same physical memory, there is no need to perform explicit data copies (e.g., `tensor.to('cuda')`). Data can be accessed by both processors without traversing a PCIe bus, which can significantly reduce latency for workloads that require tight CPU-GPU interaction.27
- **Con: Lower Peak Bandwidth and Contention.** While the UMA is high-bandwidth compared to traditional system RAM, its peak bandwidth is generally lower than the specialized HBM2 or GDDR6X memory used in high-end discrete GPUs.30 Furthermore, this is a
    
    *shared* resource. The CPU, GPU, Neural Engine, display controller, and other components all contend for access to this single memory bus.33 In a heavily utilized system, this contention can become a performance bottleneck, starving the GPU of data.
    

The UMA is a double-edged sword. Its massive capacity is a major advantage for Mamba, but its shared, lower-bandwidth nature means that performance will almost certainly be memory-bandwidth-bound, not compute-bound.32 This reinforces the importance of the kernel fusion technique discussed in Section 1; minimizing trips to main memory is paramount on this architecture.

### The GPU and Metal API: The Primary Target

The integrated GPU on the M-series chip is the workhorse for high-performance computing and the primary target for training our Mamba model. Apple provides a low-level API called Metal to program the GPU directly.34 All high-level machine learning frameworks, including TensorFlow and PyTorch, use Metal as their backend on Apple platforms.25 For the purpose of porting Mamba's custom kernel, it is useful to establish a conceptual mapping from the familiar CUDA programming model to Metal 38:

- **CUDA Grid** ↔ **Metal Grid**: The overall collection of threads for a kernel dispatch.
- **CUDA Thread Block** ↔ **Metal Threadgroup**: A group of threads that execute concurrently and can share a fast, on-chip memory space.
- **CUDA Thread** ↔ **Metal Thread**: A single execution unit.
- **CUDA `__shared__` memory** ↔ **Metal `threadgroup` memory**: The fast, on-chip memory shared by threads within a threadgroup.

This shared vocabulary will be essential when we discuss the implementation of the custom kernel in Section 3.

### The Apple Neural Engine (ANE): A Deliberate Dead End for Training

Apple's marketing materials frequently highlight the performance of the Apple Neural Engine (ANE), a dedicated co-processor for accelerating machine learning tasks.26 It is crucial for the developer to understand that the ANE is a

**red herring** for the task of training a Mamba model from scratch. Any effort invested in attempting to leverage the ANE for this project will be wasted.

This definitive conclusion is based on three key limitations of the ANE:

1. **It is an Inference Accelerator.** The ANE's architecture is optimized for high-throughput, low-power *inference* of already-trained models, not for the computationally different task of training.41 Training requires storing intermediate activations for backpropagation, a memory access pattern for which the ANE is not designed.43
2. **It has Insufficient Numerical Precision.** The ANE is primarily designed to work with low-precision data types like FP16 and INT8.41 While modern mixed-precision training heavily utilizes FP16 for performance, it critically relies on accumulating gradients and updating weights in higher-precision FP32 to avoid numerical underflow and instability.43 The ANE lacks this necessary high-precision capability for stable training.
3. **There is No Public API for Training.** There is no low-level framework for directly programming the ANE. All access is abstracted through Apple's Core ML framework, which is designed for deploying and running inference on pre-converted models.41 It does not provide the flexibility needed to define the custom forward and backward passes required to train a novel architecture like Mamba within a framework like PyTorch or MLX.

Therefore, all development effort must be focused exclusively on targeting the Apple GPU via the Metal API.

### 2.2 The Framework Dilemma: PyTorch MPS vs. Apple MLX

With the hardware target established, the next strategic decision is the software framework. There are two viable paths for implementing Mamba on Apple Silicon, each with a distinct set of trade-offs. This choice is not merely one of preference but a strategic decision between a mature but imperfect ecosystem and a native but nascent one.

### Path A - PyTorch + Metal Performance Shaders (MPS)

This is the path of least resistance for any developer coming from the broader ML ecosystem. PyTorch is the de facto standard for research, and its MPS backend provides an entry point for running models on Apple Silicon GPUs.37

- **Strengths:** The primary advantage is the vast and mature PyTorch ecosystem. The developer can leverage familiar APIs, a rich library of existing modules, sophisticated data loaders, and seamless integration with tools like Hugging Face Transformers and Weights & Biases. The entry point is simple: a call to `.to('mps')` moves a model and its data to the GPU.47
- **Weaknesses (The "Minefield"):** The MPS backend, while functional, is still in a beta phase and is known to be a "leaky abstraction" with numerous pitfalls.21 The developer must be prepared to navigate:
    - **Incomplete Operator Coverage:** Not all PyTorch operations are implemented for the MPS backend. When an unsupported op is encountered, PyTorch will silently fall back to the CPU, causing a massive performance penalty due to the data transfer and slower computation.20
    - **Performance Inconsistencies:** For some operations, the MPS implementation can be slower than a highly optimized CPU implementation, leading to counter-intuitive performance profiles.21
    - **Memory Management Bugs:** The community has reported numerous issues with memory management, including sporadic "out of memory" errors even when ample memory is available, particularly in automated environments like GitHub Actions. Tuning the `PYTORCH_MPS_HIGH_WATERMARK_RATIO` environment variable is often required but can be a process of trial and error.22
    - **Hard Limitations:** The MPS backend has a critical, hard-coded limitation: its internal kernels use 32-bit indexing, meaning they cannot operate on tensors that require addressing more than 232 bytes (4GB) of memory. This can be easily triggered by Mamba's large intermediate state and will cause the program to crash.48

### Path B - Apple MLX

MLX is Apple's own open-source array framework, designed from the ground up for high-performance machine learning on Apple Silicon.49 It presents a native, but newer, alternative to PyTorch.

- **Strengths:**
    - **Native Design:** MLX is built on the assumption of UMA. It features a unified memory model where arrays exist in shared memory without explicit device placement, simplifying code and eliminating a class of bugs.50
    - **Lazy Computation:** MLX operations build a computation graph that is only executed when a result is explicitly requested. This allows MLX's compiler to perform automatic graph-level optimizations, including operator fusion, which could improve performance even for naive implementations.50
    - **Simplified Custom Kernels:** MLX provides a much more streamlined Python API (`mlx.fast.metal_kernel`) for integrating custom Metal kernels, abstracting away the complex C++/Objective-C boilerplate required by PyTorch.52
- **Weaknesses:** The primary weakness of MLX is its youth. Its ecosystem is far smaller than PyTorch's. While it has growing support for popular models and concepts 54, the developer may need to implement more of the surrounding training infrastructure (e.g., advanced data loading, distributed training utilities, logging integrations) from scratch.

### Strategic Recommendation

This guide will detail both implementation paths, as the optimal choice depends on the project's constraints.

- **Path A (PyTorch/MPS)** is recommended for projects that require deep integration with the existing PyTorch ecosystem or for developers who wish to stay within a familiar environment. However, this path must be undertaken with the explicit understanding that achieving high performance for the `selective_scan` operation will require advancing to a complex custom Metal kernel implementation, as detailed in Section 3.
- **Path B (MLX)** is presented as a compelling alternative for new, self-contained projects where peak performance on Apple Silicon is the primary goal and the developer is willing to invest in a newer, more architecturally aligned framework. This path is detailed in Section 4.

The choice between these frameworks is a classic engineering trade-off between the maturity of a retrofitted solution and the potential of a native but nascent one. The problems with the MPS backend are largely symptoms of an architectural mismatch between PyTorch's design assumptions (discrete memory spaces) and Apple's hardware reality (UMA). MLX, by being designed for this reality, offers a potentially smoother path for hardware-specific optimization. For a project like Mamba, where the performance of a single, hardware-sensitive kernel is paramount, the framework with the lowest-friction path to custom hardware programming (MLX) may be the superior long-term choice. A prudent strategy would be to prototype the critical scan kernel in both frameworks early in the development cycle to make a data-driven decision before committing to a full training pipeline.

## Section 3: Implementation Path A: High-Performance Mamba with PyTorch & Custom Metal Kernels

This section serves as the core engineering playbook for the recommended PyTorch-based implementation. It provides a structured path that guides the developer from an initial, slow baseline to a fully optimized, high-performance solution by building a custom PyTorch operator. This process is non-trivial but essential for unlocking Mamba's true potential on Apple Silicon.

### 3.1 Baseline Implementation and Profiling: Quantifying the Bottleneck

The first step in any optimization effort is to establish a baseline and use profiling tools to identify the exact source of the performance bottleneck. This data-driven approach provides the empirical justification for the significant engineering effort of writing a custom kernel.

### Step 1: Naive Implementation

The developer should begin by implementing the Mamba block using only standard PyTorch modules. The most critical part, the `selective_scan`, will be implemented as a simple, sequential `for` loop within the `forward` method of a `torch.nn.Module`. This implementation will be functionally correct but, as predicted, inefficient.

Python

# 

```
# A simplified, naive implementation of the selective scan loop
# for demonstration purposes.
import torch

def selective_scan_naive(u, delta, A, B, C, D, z, delta_bias, h_init):
    """
    u: (B, L, D)
    delta: (B, L, D)
    A: (D, N)
    B: (B, L, N)
    C: (B, L, N)
    h_init: (B, D, N)
    """
    B, L, D = u.shape
    N = A.shape

    # Discretize A and B
    delta = torch.nn.functional.softplus(delta + delta_bias)
    delta_A = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
    delta_B_u = torch.einsum('bld,bln,bld->bldn', delta, B, u)

    h = h_init.clone()
    ys =

    for i in range(L):
        h = delta_A[:, i] * h + delta_B_u[:, i]
        y = torch.einsum('bdn,bln->bld', h, C[:, i, :].unsqueeze(2))
        ys.append(y)

    y = torch.stack(ys, dim=1) # (B, L, D)
    y = y + u * D

    return y

```

### Step 2: Move to MPS

The entire model, including all its submodules and parameters, should be moved to the `mps` device. This ensures that PyTorch attempts to execute all operations on the Apple GPU.

Python

# 

```
import torch

if not torch.backends.mps.is_available():
    raise RuntimeError("MPS backend not available.")

device = torch.device("mps")
model = MambaModel(...).to(device)
inputs = torch.randn(batch_size, seq_len, d_model, device=device)

# Run forward and backward pass
outputs = model(inputs)
loss = outputs.sum()
loss.backward()

```

### Step 3: Profile

With the naive model running on the MPS device, the next step is to profile its performance. Apple's Xcode Instruments is the primary tool for this, and PyTorch provides hooks to emit signposts that are visible within Instruments.57 The developer should wrap the model's execution with the

`torch.mps.profiler`.

Python

# 

```
# Set environment variable to see OS Signposts in Instruments
# export KINETO_USE_OS_TRACE=1

with torch.mps.profiler.profile() as prof:
    outputs = model(inputs)
    loss = outputs.sum()
    loss.backward()
    torch.mps.synchronize() # Ensure all GPU work is finished

# The profiling results can be viewed by opening the generated
#.trace file in Xcode Instruments.

```

The expected outcome of this profiling session is a clear and unambiguous result: the GPU utilization will be extremely low, and the execution timeline in Instruments will show a long sequence of very small, distinct kernel dispatches corresponding to each operation inside the Python `for` loop. The vast majority of the wall-clock time will be spent not on computation, but on the overhead of the Python interpreter dispatching these thousands of tiny kernels. This result provides the undeniable evidence that a fused, custom kernel is not an optional optimization but a fundamental necessity.

### 3.2 Engineering the Custom Selective Scan Operator in Metal

This subsection provides a detailed, step-by-step guide to building a custom PyTorch operator that calls a high-performance Metal kernel. This involves writing code in three different languages: Metal Shading Language (MSL) for the GPU kernel, C++/Objective-C for the host-side code that dispatches the kernel, and Python for the build system and final user-facing API.

### Step 1: Translating CUDA to Metal Shading Language (MSL)

The first task is to translate the algorithmic logic of the CUDA kernel into MSL. While the languages are different, the underlying GPU execution models are conceptually similar, allowing for a reasonably direct translation.38

A key translation table for GPU programming concepts is as follows:

| CUDA C++ | Metal Shading Language (MSL) | Description |
| --- | --- | --- |
| `__global__ void kernel(...)` | `[[kernel]] void kernel(...)` | Kernel function declaration |
| `blockIdx.x`, `gridDim.x` | `threadgroup_position_in_grid.x`, `grid_dim.x` | Grid and threadgroup identifiers |
| `threadIdx.x`, `blockDim.x` | `thread_position_in_threadgroup.x`, `threads_per_threadgroup.x` | Thread identifiers within a threadgroup |
| `__shared__ float tile[N];` | `threadgroup float tile[N];` | Fast, on-chip shared memory |
| `__syncthreads();` | `threadgroup_barrier(mem_flags::mem_threadgroup);` | Synchronization barrier for threads in a threadgroup |
| `__shfl_sync(...)` | `simd_shuffle(...)` | Warp/SIMD-group level data exchange |
| `float* ptr` | `device float* ptr` or `constant float* ptr` | Pointers to device (global) memory |

This translation table provides the necessary vocabulary to port the core logic.

### Step 2: Implementing the Parallel Scan in MSL

The algorithmic core of the efficient scan is a parallel prefix sum. A Blelloch-style implementation is a standard choice. It consists of two main passes:

1. **Up-Sweep (Reduction):** In this phase, threads work in a tree-like structure to compute partial sums. At each level of the tree, half the threads become inactive, while the active threads sum pairs of elements from the level below. This continues until a single thread computes the total sum for its block (threadgroup).
2. **Down-Sweep (Scan):** In this phase, the tree is traversed downwards. The partial sums computed during the up-sweep are used to construct the final prefix sum for every element in the array.

A simplified MSL snippet for a single threadgroup's worth of a parallel scan on a 1D array would look conceptually like this:

C++

# 

`#**include** <metal_stdlib>using namespace metal;

[[kernel]]
void parallel_scan_kernel(device float* data,
                          threadgroup float* shared_data,
                          uint tid [[thread_position_in_threadgroup]],
                          uint block_size [[threads_per_threadgroup]]) {
    // Load data from device to threadgroup memory
    shared_data[tid] = data[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep phase
    for (uint stride = 1; stride < block_size; stride *= 2) {
        if (tid >= stride) {
            shared_data[tid] += shared_data[tid - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    //... Down-sweep phase would follow...

    // Write result back to device memory
    data[tid] = shared_data[tid];
}`

*Note: A full, correct implementation is more complex, involving careful handling of bank conflicts and the down-sweep logic, but this illustrates the structure.*

### Step 3: Achieving Kernel Fusion in Metal

The true performance gain comes from fusing all steps of the selective scan into a single kernel. This involves expanding the simple scan kernel from Step 2 into a monolithic function that performs the entire operation described in Section 1.2.

The final `.metal` kernel's signature will be complex, accepting pointers to all the necessary input tensors:

C++

# 

`[[kernel]]
void fused_selective_scan(
    // Inputs
    device const float* x,         // (B, L, D)
    device const float* delta,     // (B, L, D)
    device const float* A,         // (D, N)
    device const float* B,         // (B, L, N)
    device const float* C,         // (B, L, N)
    device const float* D,         // (D)
    device const float* z,         // (B, L, D)
    device const float* delta_bias, // (D)
    // Output
    device float* y,               // (B, L, D)
    // Grid/thread identifiers
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    // 1. Load parameters for the current batch/dimension into threadgroup memory.
    
    // 2. Loop over the sequence length `L` in chunks that fit in threadgroup memory.
    //    For each chunk:
    //    a. Perform discretization of A and B using delta.
    //    b. Perform the parallel scan on the discretized inputs.
    //    c. Multiply the resulting state `h` by `C`.
    //    d. Add the skip connection `D*x` and apply the gating `z`.
    
    // 3. Write the final output `y` back to device memory.
}`

The internal logic of this kernel is the most complex part of the entire project, requiring careful management of threadgroup memory and synchronization to correctly implement the fused parallel scan across the sequence length.

### Step 4: The C++/Objective-C Bridge

With the Metal kernel written, a bridge is needed to call it from PyTorch. This is done using a C++ extension file that mixes C++ with Objective-C, typically with a `.mm` extension.58 This host code is responsible for the "command and control" of the GPU.

The key steps in the `.mm` file are:

1. **Receive PyTorch Tensors:** The function will be exposed to Python via `pybind11` and will accept `torch::Tensor` objects as arguments.
2. **Get Metal Device:** Obtain the default GPU device using `id<MTLDevice> device = MTLCreateSystemDefaultDevice();`.
3. **Compile the Kernel:** Load the `.metal` file from disk as a string and compile it into an `MTLLibrary` object at runtime. From this library, get a handle to the kernel function (`id<MTLFunction>`).
4. **Create Pipeline State:** Create an `id<MTLComputePipelineState>` from the kernel function. This object represents the compiled, ready-to-execute kernel.
5. **Create Metal Buffers:** For each input and output `torch::Tensor`, create a corresponding `id<MTLBuffer>`. Crucially, because of UMA, these buffers can be created to point directly to the memory already allocated by PyTorch using the tensor's `.data_ptr()`, avoiding any memory copies.
6. **Encode and Dispatch Commands:**
    - Create an `MTLCommandQueue`, `MTLCommandBuffer`, and `MTLComputeCommandEncoder`.
    - Set the pipeline state on the encoder.
    - Set each `MTLBuffer` as an argument to the kernel at the correct index.
    - Calculate the grid and threadgroup dimensions and dispatch the kernel using ``.
    - End the encoding.
7. **Execute and Synchronize:** Commit the command buffer to the queue (`) and wait for it to complete (`).
8. **Return Output:** The output tensor, which was backed by an `MTLBuffer` that the GPU wrote into, is now ready and can be returned to Python.

### Step 5: The `setup.py` and `pybind11` Wrapper

The final step is to make this C++ function callable from Python. This requires a `setup.py` file that uses `torch.utils.cpp_extension`.58 The key configurations in this file are:

- Checking for MPS availability (`torch.backends.mps.is_available()`).
- Telling the build system how to handle `.mm` files (as Objective-C++).
- Specifying the necessary compiler and linker flags, most importantly `framework Metal` and `framework Foundation` to link against Apple's frameworks.
- Defining the C++ extension using `CppExtension`, pointing to the `.mm` source file.

The `pybind11` boilerplate at the bottom of the `.mm` file exposes the C++ function to the Python module, making it importable and callable like any other Python function.

### 3.3 Field Notes on PyTorch MPS Pitfalls and Edge Cases

Even with a custom kernel for the most intensive operation, the developer will still be operating within the PyTorch MPS ecosystem, which has several known issues that require careful navigation.

### Memory Management

The `RuntimeError: MPS backend out of memory` is a common and often confusing error. It can occur even when `torch.mps.current_allocated_memory()` shows very little memory in use. This is often seen in CI/CD environments like GitHub Actions.22 The

`PYTORCH_MPS_HIGH_WATERMARK_RATIO` environment variable can be used to control the fraction of system RAM that PyTorch is allowed to use as a memory pool for the GPU. Setting it to a lower value (e.g., `0.7` or `0.6`) can sometimes prevent these errors by leaving more headroom for the OS and other processes, but it is not a guaranteed fix. The developer should be prepared for trial-and-error tuning of this parameter.

### Operator Fallbacks and Performance Cliffs

While the custom kernel handles the scan, other operations in the Mamba block (convolutions, activations, normalizations) will still rely on the native MPS backend. It is critical to ensure that none of these operations are silently falling back to the CPU. The developer can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to receive a warning whenever a fallback occurs. If a critical operation like `torch.nn.functional.silu` is not supported, the performance will be severely degraded. In such cases, the developer may need to find an alternative supported activation or, in a worst-case scenario, write another custom kernel for that specific operation.

### Numerical Stability and Mixed Precision

For optimal performance, training should be done using Automatic Mixed Precision (AMP). The developer can use `torch.amp.autocast(device_type="mps")` to enable mixed-precision training.60 However, there are several caveats:

- **Data Type Support:** Full support for `bfloat16` is a relatively recent addition and requires macOS Sonoma or later.60 On older systems, only
    
    `float16` is available for mixed precision.
    
- **NaN Propagation:** Users have reported non-deterministic behavior where `NaN` values appear randomly when running on MPS, an issue not present on CPU or CUDA.61 This can be extremely difficult to debug. Adding explicit synchronization points (
    
    `torch.mps.synchronize()`) after key operations can help isolate the source of the instability, though it comes with a performance cost. The developer must be extra vigilant about checking for `NaN`s during training.
    

### The 4GB Tensor Indexing Limit

This is arguably the most critical and insidious limitation of the MPS backend. As documented in PyTorch issue #131865, many of the underlying Metal Performance Shaders kernels use 32-bit integers for indexing, which means they cannot address elements beyond the 4GB (232 byte) boundary within a single tensor.48

For Mamba, the intermediate state h or the inputs to the internal matrix multiplications can easily exceed this size, especially with:

- Large batch sizes (B)
- Long sequences (L)
- High model dimensions (D)
- Large state dimensions (N)

When this limit is exceeded, the program will crash with an obscure error. The only robust solution is to implement **tiling** within the custom operator's host code. Before dispatching the Metal kernel, the C++ code must check the dimensions of the operation. If the total number of elements to be accessed would require an index larger than 232 (when considering element size), the operation must be broken into smaller chunks, or "tiles." For example, a large batch matrix multiply can be split along the batch dimension, and each smaller batch can be processed by a separate kernel dispatch. This is a non-trivial but absolutely necessary workaround for training large-scale Mamba models on Apple Silicon with PyTorch.

## Section 4: Implementation Path B: A Native Approach with MLX

This section explores the higher-risk, higher-reward path using Apple's native MLX framework. This path is for the developer who wishes to prioritize peak performance and architectural alignment over immediate ecosystem compatibility. MLX is designed from the ground up for Apple Silicon, and this native design offers a potentially more elegant and performant solution.

### 4.1 Re-implementing the Mamba Block in MLX

The first step is to translate the Mamba block from PyTorch to MLX. Thanks to MLX's NumPy-like API, this process is relatively straightforward for developers familiar with Python-based array programming.49 The

`mlx.nn` module provides familiar building blocks like `mlx.nn.Linear`, `mlx.nn.Conv1d`, and `mlx.nn.LayerNorm`.

A key difference from PyTorch is the absence of explicit device management. In MLX, arrays are created on the default device (which is the GPU if available), and all operations on them are automatically dispatched to the appropriate hardware. This simplifies the code by removing all `.to(device)` calls.50

Several open-source projects have already implemented Mamba in MLX, such as the one found in the `alxndrTL/mamba.py` repository.54 These can serve as excellent starting points and validation references, allowing the developer to focus on the performance-critical scan operation rather than re-implementing the entire block from scratch.

### 4.2 The MLX Selective Scan: A Simpler Path to Custom Kernels

The core challenge remains the same: the `selective_scan` must be implemented as a custom Metal kernel. However, MLX offers a dramatically simpler and more streamlined path for integrating custom kernels compared to the complex C++/Objective-C extension process required by PyTorch.

MLX provides a Python-level API, `mlx.fast.metal_kernel`, that allows a developer to write a Metal kernel as a Python string and integrate it directly into their MLX computation graph.52 This powerful API handles all the complex boilerplate that had to be written manually in Section 3.2:

- Just-in-Time (JIT) compilation of the MSL source code.
- Automatic generation of the kernel function signature based on the input arrays.
- Management of `MTLDevice`, `MTLLibrary`, and `MTLComputePipelineState`.
- Creation and management of `MTLBuffer`s.
- Encoding and dispatching the compute commands.

The developer can write the exact same high-performance, fused MSL kernel from Section 3.2.3 but can integrate it with a few lines of Python code:

Python

# 

```
import mlx.core as mx
import mlx.fast

# MSL kernel source code as a Python string
fused_scan_msl_source = """
#include <metal_stdlib>
using namespace metal;

[[kernel]]
void fused_selective_scan(...) {
    //... same kernel logic as before...
}
"""

# Create the custom kernel object once
fused_scan_kernel = mx.fast.metal_kernel(
    name="fused_selective_scan",
    input_names=,
    output_names=["y"],
    source=fused_scan_msl_source
)

# Define a Python function to call the kernel
def selective_scan_mlx(x, delta, A, B, C, D, z, delta_bias):
    B, L, D = x.shape
    #... calculate grid and threadgroup sizes...

    outputs = fused_scan_kernel(
        inputs=,
        grid=grid_dims,
        threadgroup=threadgroup_dims,
        output_shapes=[x.shape],
        output_dtypes=[x.dtype]
    )
    return outputs

```

This approach represents a major developer productivity win. It allows the engineer to focus on the logic of the high-performance kernel itself, without getting bogged down in the complexities of C++ extensions, build systems, and Objective-C interoperability.

### 4.3 Exploiting MLX-Specific Optimizations

Beyond the simplified custom kernel workflow, MLX offers several other architectural advantages that can be exploited for a Mamba implementation.

### Unified Memory Model

As mentioned, MLX's API is built on the assumption of UMA.49 This not only simplifies the code but also encourages a more efficient programming model. By treating memory as a single, unified pool, the developer is less likely to introduce unintentional data copies or transfers that can degrade performance on Apple Silicon.

### Lazy Computation and Graph Compilation

Perhaps the most powerful feature of MLX is its use of lazy computation.50 When an MLX operation is called (e.g.,

`y = mx.add(a, b)`), the computation is not immediately performed. Instead, a node is added to a computation graph. The graph is only compiled and executed when a result is explicitly requested, for example, by calling `mx.eval(y)` or trying to print a value from the array.

This has profound implications for performance. The MLX compiler can analyze the entire computation graph before execution and perform powerful optimizations, most notably **operator fusion**.51 It is possible that for the parts of the Mamba block

*outside* of the custom `selective_scan` kernel—such as the initial linear projections, the 1D convolution, and the activation functions—MLX's compiler could automatically fuse these operations into a smaller number of more efficient Metal kernels.

This suggests an interesting possibility: a "naive" Mamba implementation in MLX, even without a fully custom scan kernel, might perform significantly better than its "naive" PyTorch/MPS counterpart. The lazy evaluation and graph compilation could mitigate some of the performance penalties from using many small operators. The developer should be encouraged to experiment with this, using MLX's visualization tools (`mlx.core.export_to_dot`) to inspect the computation graph and see what fusions are being performed automatically. This could provide a significant performance boost "for free" and is a key architectural advantage over PyTorch's eager execution model on the MPS backend.

The choice of framework ultimately comes down to a trade-off between the vast ecosystem of PyTorch and the superior architectural alignment of MLX with Apple Silicon. For a project like Mamba, where the performance of a single, hardware-sensitive kernel is the primary goal, the framework that provides the lowest-friction path to custom hardware programming (MLX) is likely the superior long-term choice. The reduced complexity of kernel integration and the potential for automatic graph-level optimizations make it a compelling platform for pushing the limits of Mamba on Apple's hardware.

## Section 5: Benchmarking, Validation, and Strategic Recommendations

The final phase of the project involves rigorously measuring the performance of the implemented models, validating their correctness, and making a final, data-driven decision on the optimal implementation strategy. This section provides the tools and framework for this critical evaluation.

### 5.1 A Comparative Performance Analysis

A robust benchmarking protocol is essential to quantitatively assess the success of the optimization efforts. The protocol should measure performance across several key axes.

**Metrics:**

- **Throughput (tokens/sec):** The most important metric for training performance. This should be calculated based on the total number of tokens processed (`batch_size * sequence_length`) divided by the wall-clock time for a full forward and backward pass.
- **Latency (ms):** The wall-clock time for a single forward and backward pass. This is useful for understanding the responsiveness of the model.
- **Peak Memory Usage (GB):** The maximum amount of unified memory consumed during the training step. This should be monitored using Xcode Instruments.
- **GPU Utilization (%):** The percentage of time the GPU's execution units are active. High utilization indicates that the GPU is not being starved of data or bottlenecked by dispatch overhead. This can also be measured with Xcode Instruments.

Methodology:

The benchmark should be run across a range of sequence lengths (e.g., 1024, 4096, 16384, 65536) to test the model's linear scaling properties. For each configuration, multiple warm-up iterations should be run before starting the measurement to ensure that any JIT compilation or caching effects do not skew the results. The final reported number should be the average of several measured iterations.

Key Table to Generate:

The results of this analysis should be compiled into a comprehensive table. This table is the ultimate deliverable of the performance analysis, providing a clear, quantitative comparison of all implemented approaches. It serves as the data-driven foundation for the final strategic recommendations.

| Implementation Path | Sequence Length | Precision | Throughput (tokens/sec) | Peak Memory (GB) | GPU Utilization (%) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| CPU Baseline | 1024 | FP32 | *Value* | *Value* | 0% | Establishes the performance floor. |
| Naive PyTorch/MPS | 1024 | FP32 | *Value* | *Value* | *Value* | Expected to be slow due to I/O and overhead. |
| **Optimized PyTorch/Metal** | 1024 | FP32 | *Value* | *Value* | *Value* | **High-performance path A.** |
| **Optimized PyTorch/Metal** | 1024 | AMP (FP16) | *Value* | *Value* | *Value* | Measures benefit of mixed precision. |
| **Optimized MLX/Metal** | 1024 | FP32 | *Value* | *Value* | *Value* | **High-performance path B.** |
| **Optimized MLX/Metal** | 1024 | BF16 | *Value* | *Value* | *Value* | Measures benefit of mixed precision. |
| *... (repeat for 4096, 16384, etc.)* | ... | ... | ... | ... | ... | ... |

This table provides a powerful narrative. The "Naive PyTorch/MPS" result will quantitatively demonstrate the "Performance Chasm" discussed in Section 1, justifying the need for a custom kernel. The two "Optimized" paths will show a dramatic performance increase, proving the effectiveness of the custom Metal kernel. Finally, the direct comparison between the optimized PyTorch and MLX implementations will provide the definitive data for the developer to choose the best path forward, resolving the trade-off between ecosystem maturity and architectural alignment.

### 5.2 Validation and Advanced Debugging

Performance is meaningless if the results are incorrect. Rigorous validation is a non-negotiable step.

### Numerical Equivalence

The output of the custom Metal kernel must be validated against a trusted reference implementation. The developer should run the same set of random inputs through both the original Mamba implementation (running on CPU or, if available, CUDA) and the new Metal implementation. The outputs should be compared using a small tolerance to account for floating-point arithmetic differences (`torch.allclose` is the standard tool for this). This check should be performed for both the forward pass and, critically, for the gradients computed during the backward pass (`tensor.grad`). Any significant divergence indicates a bug in the kernel logic.

### Advanced Metal Debugging

When performance is not as expected or when bugs are difficult to trace, Xcode Instruments provides powerful tools for deep-diving into the GPU's execution.62 The

**Metal System Trace** template is particularly valuable. It allows the developer to:

- Visualize the timeline of all Metal command buffers and kernel dispatches.
- Analyze the execution time of individual shaders.
- Inspect memory allocations and traffic on the UMA bus.
- Identify GPU pipeline stalls or other low-level performance issues.

Mastering this tool is essential for fine-tuning the custom kernel, for example, by adjusting threadgroup sizes to maximize occupancy or reorganizing memory access patterns to improve cache locality.

### 5.3 Final Verdict and Future Outlook

This guide has laid out two viable paths for achieving high-performance Mamba training on Apple Silicon. The final choice depends on the specific constraints and goals of the project. A decision can be framed using the following matrix:

| Factor | Path A: PyTorch + Custom Metal | Path B: MLX + Custom Metal |
| --- | --- | --- |
| **Ecosystem Maturity** | **High.** Access to the full PyTorch ecosystem. | **Low.** Newer framework, requires more custom infrastructure. |
| **Time to Initial Prototype** | **Fast.** A naive MPS model is quick to write. | **Fast.** A naive MLX model is also quick to write. |
| **Custom Kernel Complexity** | **High.** Requires complex C++/Objective-C extension. | **Low.** Simple, direct Python API for custom kernels. |
| **Peak Performance Potential** | **High.** Limited by MPS backend overhead and bugs. | **Very High.** Architecturally aligned, potential for graph fusion. |
| **Long-Term Maintainability** | **Medium.** Depends on PyTorch's MPS support roadmap. | **High.** As a first-party Apple framework, it is the strategic direction. |

**Recommendation:**

- For projects that **must** integrate with existing PyTorch-based infrastructure, **Path A** is the necessary choice. The developer must budget significant time for the complex custom kernel integration and for debugging the idiosyncrasies of the MPS backend.
- For new, self-contained projects where maximizing performance on Apple Silicon is the primary objective, **Path B** is the strongly recommended path. The lower barrier to custom kernel development and the superior architectural alignment of MLX make it a more direct and potentially more performant route to success.

The landscape of machine learning on Apple Silicon is evolving rapidly. Apple continues to invest heavily in its ML stack, with annual improvements to Metal, Core ML, and frameworks like MLX.31 It is likely that the capabilities and performance of both the PyTorch MPS backend and the MLX framework will continue to improve. However, the fundamental principles of hardware-aware algorithm design—kernel fusion, parallelization, and intelligent memory management—will remain constant. By mastering these principles, as detailed in this guide, the developer will be well-equipped to not only implement Mamba successfully today but also to tackle the high-performance deep learning challenges of tomorrow on this unique and powerful platform.