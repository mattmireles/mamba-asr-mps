# **Field Guide: Implementing and Profiling Mamba-Style Sequence Models on Apple Silicon**

April 2, 2026

## **1\. Executive Summary**

Deploying state-space models (SSMs) and selective scan operations natively on Apple Silicon requires navigating an ecosystem characterized by an immature compilation stack, strict memory contiguity requirements, and the unique physics of a unified memory architecture. For engineers actively maintaining and shipping the Mamba-ASR-MPS branch and related automatic speech recognition (ASR) pipelines, traditional CUDA-based optimization mentalities are counterproductive. The following operational realities form the foundation of this field guide:

* **CUDA Parity is Non-Existent:** Custom Triton kernels and highly optimized CUDA C++ binaries fundamental to the original Mamba implementation do not map to Apple Silicon natively.1 Direct conversions using sequential pseudo-parallel loops completely destroy GPU throughput.  
* **Log-Space Formulations are Mandatory:** The only viable pure PyTorch parallel scan on the Metal Performance Shaders (MPS) backend relies on transforming the recurrence into a parallel prefix sum via log-space (torch.log followed by torch.cumsum and torch.exp). This achieves a 3x speedup over sequential loops.2  
* **Memory Strides Dictate Correctness:** Historically, the MPS backend fails silently on non-contiguous tensors. In-place operations (e.g., addcmul\_) on strided memory often skip execution, silently corrupting the sequence state without throwing an exception.3  
* **FP16 Accumulation is Fatal:** While mixed-precision execution is highly desirable for unified memory bandwidth, the cumsum operation in the selective scan will rapidly overflow in FP16 over long sequences (e.g., \>4096 tokens). Establishing strict FP32 islands for the scan phase is non-negotiable.4  
* **Synchronization Hides the Bottleneck:** The MPS backend operates asynchronously. Profiling without strictly enforcing torch.mps.synchronize() merely measures the Python kernel launch time, obscuring the severe dispatch overhead present on small tensor operations.7  
* **Micro-Kernel Dispatch Overhead:** Apple Silicon suffers from disproportionate kernel launch overhead relative to its raw compute capacity for micro-operations. Fusing kernels or avoiding overly fragmented operations is critical to achieving the theoretical linear-time scaling of SSMs.9  
* **Unified Memory is the Primary Weapon:** The shared RAM pool between the CPU and GPU (up to 192GB on high-end hardware) eliminates the PCIe bottleneck. This fundamentally alters the boundaries of long-sequence execution, allowing Mac hardware to bypass the standard VRAM limitations that constrain discrete GPUs.11  
* **Page Faults Destroy Throughput:** Operating too close to the unified memory limit triggers swap space utilization. The resulting page faulting degrades performance exponentially and destabilizes the training loop.11  
* **ASR Frontend Receptive Fields Must Compress:** When integrating Mamba blocks downstream of a Convolutional Neural Network (CNN) frontend, the CNN's stride parameters must aggressively compress the raw audio waveform into dense representations (e.g., 20ms frames) to prevent the linear-time SSM from processing redundant, microscopic acoustic states.13  
* **Encoder-Only ASR Fits Best:** Mamba integrates seamlessly into encoder-only ASR setups (e.g., HuBERT, Wav2Vec2 variants), whereas encoder-decoder architectures require complex cross-scan mechanisms that are currently hostile to the MPS graph compiler.15  
* **Fallbacks are Viable but Leaky:** For esoteric operations lacking native MPS bindings, wrapping the operation via torch.library.custom\_op to force a CPU fallback is safer than writing experimental Metal kernels. However, frequent context switching heavily fragments the Objective-C autorelease pool, causing slow memory leaks.17  
* **Dynamo/Inductor Gaps:** PyTorch's torch.compile provides marginal to negative acceleration on MPS due to the lack of mature backend support. The Inductor compiler routinely graph-breaks on dynamic audio sequence lengths.19  
* **The Framework Divergence:** While the MLX framework is vastly superior for Apple Silicon edge inference (up to 3x faster), PyTorch's MPS backend remains highly competitive for specific training passes, particularly vector-Jacobian products crucial for backpropagation.11

## **2\. Decision Framework**

Architectural decisions regarding the deployment of Mamba-style models on Apple Silicon hinge entirely upon sequence length requirements, acceptable latency, and the deployment target's OS version. Engineers should consult the following framework before merging sequence-modeling updates into the Mamba-ASR-MPS repository.

### **When Mamba/Selective Scan is Worth Pursuing on Apple Silicon**

Pursue native, pure PyTorch log-space parallel scans on MPS when the audio sequence length—post-frontend compression—exceeds 4096 tokens, and the training or deployment target runs macOS 14.x or newer. At this length, the ![][image1] linear scaling of the selective scan decisively overcomes the high kernel dispatch overhead of the MPS backend.22 This approach relies heavily on torch.cumsum, which is natively supported and heavily optimized within the MPSGraph framework, offering the most stable path forward for long-form speech transcription.2

### **When Standard Transformer Components are Simpler and Better**

If the ASR sequence length remains strictly under 1024 tokens, standard Multi-Head Attention (MHA) utilizing highly optimized FlashAttention-style primitives provided natively by Apple's Metal Performance Shaders is significantly faster. The operational complexity of maintaining numerical stability within a selective scan for short sequences rarely justifies the latency gains.23 Apple's Matrix Cores (AMX) are highly tuned for the dense matrix multiplications inherent to Transformers, easily outpacing the sequential memory-bound nature of unoptimized SSM kernels at short lengths.11

### **When Hybrid Fallback is Acceptable**

Hybrid fallback (routing specific operations to the CPU) is a valid, sane path under two precise conditions:

1. **Gating and Control Flow:** The tensor shapes are minuscule, and the operation determines dynamic control flow where GPU synchronization would stall the pipeline.  
2. **Missing MPS Operators:** The architecture relies on specific sorting algorithms, complex numbers, or experimental SSM recurrences that trigger a "NotImplementedError" on MPS.17 The unified memory architecture mitigates the traditional PCIe bottleneck, making CPU fallbacks computationally viable on Mac hardware.11

### **When CoreML or MLX is a Better Use of Time**

If the sole objective of the branch is real-time edge inference on consumer MacBooks with rigid latency budgets, optimizing the PyTorch MPS training code for inference is a wasted effort. Exporting the trained weights to MLX provides up to a 3x throughput increase out-of-the-box due to lower Python overhead and native unified memory optimization.20

### **Hardware Boundary Matrix**

The following table outlines the operational boundaries dictating framework and architecture choices for the Mamba-ASR-MPS branch based on hardware limits.

| Sequence Length (Tokens) | Apple Memory Limit (Unified) | Optimal Architecture | Optimal Framework | Primary Bottleneck |
| :---- | :---- | :---- | :---- | :---- |
| \< 1,024 | 16GB \- 32GB | Transformer / Dense Attention | PyTorch (MPS) | Apple Kernel Dispatch Overhead |
| 1,024 \- 4,096 | 32GB \- 64GB | Hybrid (ConMamba / Jamba) | PyTorch (MPS) | Tensor Contiguity & Casting |
| 4,096 \- 16,384 | 64GB \- 128GB | Pure Mamba (Log-Space Scan) | PyTorch (MPS) | FP16 Prefix Sum Overflow |
| \> 16,384 | \> 128GB (Mac Studio) | Pure Mamba (Log-Space Scan) | MLX (Inference) | Page Faults & OS Swap Thrashing |

## **3\. Migration Playbook**

Introducing a Mamba-style block into an existing sequence model pipeline requires an exact, phased approach to prevent silent numerical degradation on the Apple Silicon backend.

### **Phase 1: Audit Checklist for Introducing a Mamba-Style Block**

* **Dependency Isolation:** Eradicate all mamba\_ssm binaries compiled with nvcc or Triton from the environment. Ensure the branch utilizes a pure PyTorch implementation compatible with MPS.1  
* **Device Agnosticism:** Verify all tensor instantiations (e.g., torch.zeros, torch.arange) explicitly inherit the .device property of the input tensor. Hardcoding 'cuda' will crash the initialization script.5  
* **Contiguity Audit:** Map every tensor that undergoes an in-place mutation (add\_, addcmul\_, copy\_). Ensure an .is\_contiguous() check exists immediately prior to the mutation.3

### **Phase 2: Ordered Implementation Steps**

1. **Strip the Pseudo-Parallel Loops:** Remove any for t in range(seq\_len): constructs utilized for the selective scan. While functional, they force the GPU to process elements sequentially, destroying throughput.2  
2. **Integrate the Log-Space Scan:** Replace the loops with the mathematically equivalent torch.cumsum(torch.log(A.clamp(min=1e-20)), dim=1) formulation to achieve true hardware parallelism.2  
3. **Establish Precision Islands:** Cast the ![][image2] and ![][image3] state tensors to torch.float32 immediately prior to the log-space transformation to prevent gradient underflow, returning to torch.float16 or torch.bfloat16 upon completion.4  
4. **Integrate the CNN Frontend:** Connect the 1D CNN acoustic feature extractor. Ensure the CNN output is transposed appropriately (Batch, Seq\_Len, Channels) before feeding into the linear Mamba block.27

### **Phase 3: Validation Checklist After Each Step**

* **Token-for-Token Equality:** The output of the MPS Mamba block must match a strict CPU reference implementation up to a 1e-4 absolute tolerance over a 64-step decoding pass.19  
* **Gradient Flow Verification:** Verify that exp\_avg\_sq inside the AdamW optimizer updates properly across all layers. Silent stride failures on MPS often manifest as frozen gradients in deep layers rather than immediate crashes.3  
* **Dispatch Overhead Measurement:** Profile the execution using torch.mps.synchronize(). Ensure the kernel launch time does not vastly exceed the computation time for short audio segments.8

## **4\. Failure Modes Catalog**

This catalog details the critical failure modes encountered when optimizing and debugging selective scan kernels on Apple Silicon, organized by the specific symptom observed during execution.

### **Failure Mode 1: Silent Output Failures on Strided Tensors**

* **Symptom:** The loss plateaus mysteriously at the beginning of training. Upon inspection, specific state matrices remain populated with zeroes, or gradients fail to update, despite forward passes completing without triggering any PyTorch exceptions.  
* **Root Cause:** Prior to macOS 15, the MPSGraph backend lacked comprehensive support for strided array views. Operations relying on in-place mutation (e.g., addcmul\_, addcdiv\_, copy\_) on non-contiguous tensors fail to execute the underlying Metal kernel. The CPU dispatches the command, but the GPU skips the operation, leaving the output buffer entirely unmodified.3  
* **Minimal Fix:** Blindly append .contiguous() to any tensor before it undergoes an in-place mutation.  
* **Robust Fix:** Implement a generalized wrapper that asserts contiguity for all inputs entering the SSM block, paying the memory reallocation copy cost strictly once at the module boundary rather than deep within the recurrence loop.3  
* **Code Example:**  
  Python  
  import torch

  def safe\_state\_update\_mps(hidden\_state: torch.Tensor, delta: torch.Tensor) \-\> torch.Tensor:  
      """  
      Safely updates the hidden state, ensuring MPS backend does not silently drop  
      the computation due to strided memory layouts.  
      """  
      \# WORST PRACTICE: hidden\_state.addcmul\_(delta, other\_tensor)

      \# ROBUST FIX: Enforce memory contiguity before kernel dispatch  
      if not hidden\_state.is\_contiguous():  
          hidden\_state \= hidden\_state.contiguous()  
      if not delta.is\_contiguous():  
          delta \= delta.contiguous()

      \# Perform out-of-place or safe in-place operation  
      hidden\_state \= hidden\_state \+ (delta \* hidden\_state)  
      return hidden\_state

* **Verification:** Inspect the tensor's metadata via tensor.is\_contiguous(). Print the sum() of the tensor immediately before and after the operation to verify that the mathematical mutation physically occurred on the device.

### **Failure Mode 2: FP16 Overflow in Prefix Sums**

* **Symptom:** NaN values suddenly propagate through the sequence dimension halfway through a long audio segment (e.g., \>8192 tokens), completely corrupting the loss.  
* **Root Cause:** The cumulative sum (torch.cumsum) operation in the selective scan exponentially compounds values. When executed in FP16 or BF16, the limited dynamic range of the exponent bits causes rapid overflow to infinity or underflow to zero. Because the recurrence state ![][image4] relies on the cumulative sum of ![][image2], the state vector instantly collapses.4  
* **Minimal Fix:** Cast all inputs to float32 globally, sacrificing VRAM and bandwidth for stability.  
* **Robust Fix:** Implement a precision-island context manager. Force FP32 accumulation specifically for the log, cumsum, and exp operations of the selective scan, while keeping the heavy parameter matrix multiplications (GEMMs) in FP16.  
* **Code Example:**  
  Python  
  import torch

  def log\_space\_scan\_fp32\_island(x: torch.Tensor, A: torch.Tensor) \-\> torch.Tensor:  
      """  
      Executes the selective scan within an FP32 island to prevent exponent overflow  
      on Apple Silicon during long-sequence cumulative sums.  
      """  
      orig\_dtype \= x.dtype \# Usually float16 or bfloat16

      \# 1\. Enter FP32 Island  
      x\_fp32 \= x.to(torch.float32)  
      A\_fp32 \= A.to(torch.float32)

      \# 2\. Prevent log(0) with extreme clamping  
      log\_A \= torch.log(A\_fp32.clamp(min\=1e-20))

      \# 3\. Parallel scan via cumsum in float32  
      \# Adding x\_fp32 safely without exceeding exponent limits  
      scan\_result \= torch.cumsum(log\_A.unsqueeze(0).unsqueeze(0) \+ x\_fp32, dim=1)

      \# 4\. Exponential recovery and cast back to original precision  
      return torch.exp(scan\_result).to(orig\_dtype)

* **Verification:** Pass a 16,000-token sequence block populated entirely with ones through the block. Ensure the final token's state vector contains finite numbers and does not evaluate to torch.isnan().

### **Failure Mode 3: Dispatch Overhead Dominating Short Sequences**

* **Symptom:** The Mamba block performs significantly slower than a standard Multi-Head Attention block on short audio sequences (\<128 tokens) despite possessing a theoretical linear-time ![][image1] advantage.  
* **Root Cause:** Apple's Metal API incurs a kernel launch and dispatch overhead. For short sequences, the time taken by the Python CPU thread to dispatch the multiple sequential operations of the state space model exceeds the actual GPU compute time required to execute them.9  
* **Minimal Fix:** Pad and batch smaller audio segments together to increase the overall compute workload per kernel dispatch, hiding the latency.  
* **Robust Fix:** Fuse operations into larger blocks using AI-generated Metal kernels if available, or rely on higher-level composite PyTorch operators. Do not split operations into dozens of microscopic tensor multiplications.  
* **Verification:** Compare the execution time of batch size 1 against batch size 32 using torch.mps.synchronize(). If batch size 32 takes identical total time to batch size 1, the pipeline is entirely bottlenecked by dispatch overhead, not the ALUs.9

### **Failure Mode 4: ASR CNN Frontend Stride Mismatch**

* **Symptom:** The Mamba model converges during training, but the Word Error Rate (WER) remains exceptionally high. The model hallucinates rapid, repeating phonemes or fails to capture long-term acoustic context.  
* **Root Cause:** The temporal resolution of the CNN frontend does not match the receptive capacity of the downstream SSM. Unlike Transformers, which possess global attention and can view the entire sequence simultaneously, SSMs ingest data autoregressively. If the CNN stride is too small (e.g., producing 5ms frames), the sequence length explodes into the tens of thousands for short audio clips. The SSM's hidden state capacity is overwhelmed, and it "forgets" distant acoustic context.13  
* **Minimal Fix:** Increase the stride of the 1D convolutions in the frontend to reduce the frame rate (e.g., from 10ms to 20ms frames).  
* **Robust Fix:** Implement a hierarchical convolutional feature extractor that heavily downsamples the raw 16kHz waveform, outputting dense embeddings that align with standard phoneme duration before handing the sequence to the Mamba block.28  
* **Code Example:**  
  Python  
  import torch  
  import torch.nn as nn

  class SpeechFrontendMamba(nn.Module):  
      def \_\_init\_\_(self, input\_channels=1, out\_channels=256):  
          super().\_\_init\_\_()  
          \# Objective: Downsample raw 16kHz audio to prevent SSM context bloat.  
          \# Stride of 160 \= 10ms frames. Stride of 320 \= 20ms frames.  
          self.cnn\_extractor \= nn.Sequential(  
              nn.Conv1d(input\_channels, 64, kernel\_size=400, stride=160, padding=200),  
              nn.BatchNorm1d(64),  
              nn.GELU(),  
              \# Secondary downsampling to achieve \~40ms conceptual frames  
              nn.Conv1d(64, out\_channels, kernel\_size=3, stride=2, padding=1)  
          )

      def forward(self, waveform: torch.Tensor) \-\> torch.Tensor:  
          \# waveform: (Batch, 1, Time\_samples)  
          features \= self.cnn\_extractor(waveform) \# Output: (Batch, Channels, Time\_frames)

          \# Transpose for SSM consumption: (Batch, Time\_frames, Channels)  
          return features.transpose(1, 2).contiguous() 

* **Verification:** Validate the tensor shape exiting the CNN. A 1-second audio clip at 16kHz should ideally produce a sequence length dimension strictly between 25 and 50 before entering the Mamba block.

### **Failure Mode 5: CPU Fallback Memory Leaks**

* **Symptom:** System Resident Set Size (RSS) unified memory climbs continuously during training until the OS terminates the process via the Out-Of-Memory (OOM) killer.  
* **Root Cause:** When relying on torch.library.custom\_op to route unsupported operations to the cpu\_fallback, the constant context switching between the CPU and GPU heavily fragments macOS's Objective-C++ autoreleasepool. The OS fails to garbage-collect the temporary tensors generated during the fallback transition, causing a slow memory leak.18  
* **Minimal Fix:** Insert torch.mps.empty\_cache() periodically at the end of the training loop.  
* **Robust Fix:** Avoid bridging tensors back and forth repeatedly within a single block. If a fallback is necessary, execute a continuous block of operations sequentially on the CPU before returning the final resulting tensor to the MPS device.  
* **Verification:** Monitor memory utilization using asitop or the macOS Activity Monitor. The memory usage graph should stabilize into a flat line after the first two training epochs, rather than climbing linearly.

### **Failure Mode 6: Page Fault Death on Unified Memory Limits**

* **Symptom:** Training begins at 2 iterations per second. Halfway through the epoch, performance plummets to 0.05 iterations per second. The system becomes completely unresponsive.  
* **Root Cause:** Apple Silicon utilizes a Unified Memory Architecture (UMA). When PyTorch allocates memory approaching the physical limit of the machine (e.g., 60GB on a 64GB machine), the operating system aggressively swaps memory to the SSD. The resulting page faulting destroys throughput and halts the GPU.11  
* **Minimal Fix:** Hardcode a lower batch size or drastically reduce the maximum audio sequence length.  
* **Robust Fix:** Use torch.mps.set\_per\_process\_memory\_fraction() to artificially cap PyTorch's memory usage to 85% of physical RAM, forcing PyTorch to raise a clean RuntimeError: Out of Memory instead of crashing the host OS via page faults.34

## **5\. Best Practices**

### **Exploit the Unified Memory Architecture**

Apple Silicon's unified memory implies that moving data between the CPU and GPU does not traverse a slow PCIe bus; it is essentially a pointer handoff. For massive sequence lengths that exceed traditional VRAM boundaries, engineers can actively choose to hold specific cache states in CPU memory without incurring the severe latency penalties seen on NVIDIA architectures.11

### **Utilize the Log-Space Formulation for Parallel Scans**

Pure PyTorch implementations must use the true parallel scan via log-space. The traditional approach of looping over sequence lengths (pseudo-parallelism) runs sequentially on the GPU, completely starving the compute cores. By clamping the matrix ![][image2], taking the logarithm, performing a native torch.cumsum, and recovering via torch.exp, the implementation transforms a highly sequential recurrence into a parallel prefix sum matrix operation natively supported by MPSGraph.2

### **Isolate the MPS Profiling Context**

Timing code on Apple Silicon requires aggressive synchronization. Since the MPS backend operates asynchronously, measuring the time delta utilizing standard Python time functions without torch.mps.synchronize() will only measure the Python dispatch time, yielding falsely optimistic results. Always bracket performance-critical code with synchronization barriers to extract ground-truth metrics.8

### **Modularize ASR Components: Encoder-Only vs Encoder-Decoder**

Separate the acoustic frontend from the sequence model. For ASR, **Encoder-Only** architectures (like HuBERT or Wav2Vec2) paired with a Mamba backbone represent the gold standard for Apple Silicon. Mamba is exceptionally efficient at extracting features linearly over the audio. Conversely, **Encoder-Decoder** setups require cross-attention or cross-scan mechanisms linking the acoustic representation to the text generation phase, which currently maps poorly to the MPS backend due to complex dynamic shapes and broadcasting errors.15

## **6\. Worst Practices / Anti-Patterns**

### **Anti-Pattern: Frequent Device Context Switching**

While unified memory minimizes the transfer latency, constantly switching a tensor .to('cpu') and .to('mps') within a single forward pass forces the MPS Graph compiler to shatter the computational graph. This breaks any potential for automatic kernel fusion under the hood and introduces severe scheduling overhead that destroys training throughput.

### **Anti-Pattern: Unnecessary Contiguity Enforcement**

While Failure Mode 1 highlighted the dangers of non-contiguous tensors, blindly appending .contiguous() to every tensor operation creates massive, redundant memory allocations. Copying a 100MB sequence tensor 14 times per forward pass will heavily degrade throughput. Contiguity checks should only occur at the boundary of a custom kernel, an in-place mutation, or when explicitly viewing a transposed matrix.3

### **Anti-Pattern: Assuming PyTorch compile Resolves Inefficiencies**

Using torch.compile on Apple Silicon does not yield the same massive speedups as it does via Triton on CUDA architectures. The Inductor backend for MPS is still heavily maturing, and many complex state-space operations will trigger graph breaks or data-dependent errors (DDEs). The resulting compiled execution is frequently slower than eager mode.19 Manual optimization of the operators remains absolutely necessary.

### **Anti-Pattern: Using Pseudo-Parallel Scans for Production**

Implementing the selective scan as a Python for loop over the sequence dimension is acceptable for validation but is an extreme anti-pattern for shipping code. It restricts the GPU to sequential execution, resulting in training times 3x to 5x slower than the log-space parallel approach.2

## **7\. Weird Patterns That Work Surprisingly Well on Apple Silicon**

### **The MLX/PyTorch Dual-Framework Strategy**

Due to current ecosystem optimizations, PyTorch's MPS backend handles autograd and complex neural network training routines (particularly Vector-Jacobian products) significantly faster than MLX. Conversely, MLX excels at low-latency inference and deployment on edge devices.11 A highly successful, albeit unorthodox, pattern involves training the Mamba-ASR model entirely in PyTorch, exporting the final weights, and writing a clean MLX inference script for final edge deployment.

### **CPU Offloading for Small Attention Islands**

If the architecture utilizes hybrid Mamba-Attention blocks (e.g., Jamba-style models), routing extremely small attention matrices to the CPU while the GPU handles the massive SSM scan can sometimes yield higher total throughput. The CPU on M-series chips is exceptionally fast at handling small, irregular control flows, and unified memory makes the handoff nearly free.12

### **Transcendent Mathematics for Basic Scans**

In standard numerical computing, constantly switching between linear and logarithmic space is heavily avoided due to the computational expense of transcendental functions. On Apple Silicon, the GPU ALUs process log and exp at a high enough rate that utilizing them to mathematically unroll a recursive sequence into a parallel cumsum yields a massive net performance positive.2

## **8\. Limitations and Unsolved Problems**

### **Hardware-Level Tensor Core Incompatibilities**

NVIDIA architectures leverage specific Tensor Cores to accelerate matrix multiplications (GEMMs) in FP16/BF16. Apple Silicon's Matrix Cores (AMX) operate under fundamentally different constraints and are not always automatically targeted by PyTorch's MPS backend for arbitrary tensor shapes.11 Consequently, the theoretical peak FLOPs difference between matmul and non-matmul operations behaves differently on Mac hardware, altering optimal algorithmic design compared to the original Mamba paper.24

### **The Missing Triton Ecosystem**

The largest unsolved problem for Mamba on Apple Silicon is the lack of a mature intermediate representation compiler. The original Mamba architectures rely heavily on custom Triton kernels that map directly to NVIDIA hardware.40 Apple's Metal backend has no direct Triton equivalent. Consequently, engineers are forced to write raw Metal Shading Language (MSL) or rely on higher-level PyTorch primitives, severely capping the absolute ceiling of hardware utilization.19

### **Dynamic Shape Tracing and Data-Dependent Errors (DDEs)**

When handling variable-length audio sequences across a speech dataset, PyTorch's internal tracing mechanisms often trigger data-dependent errors (DDEs) or force continuous graph recompilations. While PyTorch 2.5+ has improved unbacked shape semantics, dynamic audio lengths still frequently cause overhead spikes when tracing custom operators on the MPS backend, rendering dynamic batching highly inefficient.36

## **9\. Copy-Paste Reference Snippets**

The following code snippets are hardened specifically for the Mamba-ASR-MPS deployment environment.

### **Snippet 9.1: Safe Selective-Scan Log-Space Wrapper**

This function replaces the CUDA-dependent selective scan with an MPS-compatible, numerically stable log-space parallel scan.

Python

import torch

def mps\_selective\_scan(  
    x: torch.Tensor,   
    delta: torch.Tensor,   
    A: torch.Tensor,   
    B: torch.Tensor,   
    C: torch.Tensor  
) \-\> torch.Tensor:  
    """  
    Computes the selective scan using parallel log-space accumulation.  
    Optimized for Apple Silicon (MPS).  
      
    Args:  
        x: (batch, seq\_len, d\_inner)  
        delta: (batch, seq\_len, d\_inner)  
        A: (d\_inner, d\_state)  
        B: (batch, seq\_len, d\_state)  
        C: (batch, seq\_len, d\_state)  
    Returns:  
        y: (batch, seq\_len, d\_inner)  
    """  
    batch, seq\_len, d\_inner \= x.shape  
      
    \# 1\. Enforce contiguity for MPS stability  
    if not x.is\_contiguous(): x \= x.contiguous()  
    if not delta.is\_contiguous(): delta \= delta.contiguous()  
      
    \# 2\. FP32 precision island for accumulation to prevent overflow  
    orig\_dtype \= x.dtype  
    delta\_f \= delta.to(torch.float32)  
    A\_f \= A.to(torch.float32)  
      
    \# 3\. Discretize A (delta \* A)  
    \# delta\_A shape: (batch, seq\_len, d\_inner, d\_state)  
    delta\_A \= torch.einsum('bld,dn-\>bldn', delta\_f, A\_f)  
      
    \# 4\. Log-space conversion with extreme clamping to prevent log(0)  
    log\_delta\_A \= torch.log(delta\_A.clamp(min\=1e-20))  
      
    \# 5\. Parallel scan via cumsum (the core Apple Silicon optimization)  
    scan\_A \= torch.cumsum(log\_delta\_A, dim=1)  
    exp\_scan\_A \= torch.exp(scan\_A)  
      
    \# 6\. Compute state updates  
    \# delta\_B\_x shape: (batch, seq\_len, d\_inner, d\_state)  
    delta\_B\_x \= torch.einsum('bld,bln,bld-\>bldn', delta\_f, B.to(torch.float32), x.to(torch.float32))  
      
    \# Weight by the inverse of the accumulated A  
    weighted\_updates \= delta\_B\_x / exp\_scan\_A.clamp(min\=1e-20)  
      
    \# Accumulate updates over the sequence  
    accumulated\_states \= torch.cumsum(weighted\_updates, dim=1)  
      
    \# Final hidden states reconstruction  
    hidden\_states \= accumulated\_states \* exp\_scan\_A  
      
    \# 7\. Project to output using C  
    y \= torch.einsum('bldn,bln-\>bld', hidden\_states, C.to(torch.float32))  
      
    return y.to(orig\_dtype)

### **Snippet 9.2: Correctness Validation Harness**

A strict validation suite to guarantee the MPS implementation has not drifted from the CPU reference implementation.

Python

import torch

def validate\_mps\_correctness(module, batch\_size=2, seq\_len=1024, d\_model=256):  
    """  
    Validates a Mamba block's output on MPS against a CPU reference implementation.  
    Crucial for catching silent stride failures or FP16 overflows.  
    """  
    print("Initiating MPS Correctness Validation...")  
      
    \# Create identical inputs  
    cpu\_input \= torch.randn(batch\_size, seq\_len, d\_model, requires\_grad=True)  
    mps\_input \= cpu\_input.detach().clone().to('mps').requires\_grad\_(True)  
      
    \# Initialize identical models  
    cpu\_model \= module(d\_model=d\_model).to('cpu')  
    mps\_model \= module(d\_model=d\_model).to('mps')  
      
    \# Force identical weights  
    mps\_model.load\_state\_dict(cpu\_model.state\_dict())  
      
    \# Forward Pass execution  
    cpu\_output \= cpu\_model(cpu\_input)  
    mps\_output \= mps\_model(mps\_input)  
      
    \# Calculate difference  
    max\_diff \= torch.max(torch.abs(cpu\_output \- mps\_output.cpu())).item()  
      
    print(f"Max Absolute Difference: {max\_diff:.6f}")  
    if max\_diff \> 1e-4:  
        print("\[\!\] RED FLAG: Divergence detected beyond acceptable FP16 tolerance.")  
        return False  
          
    print("\[✓\] Forward pass verified. MPS output matches CPU.")  
    return True

### **Snippet 9.3: Long-Sequence MPS Profiling Harness**

A standardized benchmarking tool incorporating necessary Metal synchronizations to prevent false-positive dispatch timings.

Python

import time  
import torch

def profile\_long\_sequence\_mps(model, seq\_length, iterations=10):  
    """  
    Measures true execution time of a sequence model on MPS,   
    accounting for asynchronous dispatch overhead.  
    """  
    device \= torch.device('mps')  
    model \= model.to(device)  
    dummy\_audio \= torch.randn(1, 1, seq\_length, device=device)  
      
    print(f"Profiling sequence length: {seq\_length}")  
      
    \# Warmup: Initialize kernel caches  
    for \_ in range(3):  
        \_ \= model(dummy\_audio)  
    torch.mps.synchronize()  
      
    start \= time.perf\_counter()  
    for \_ in range(iterations):  
        \_ \= model(dummy\_audio)  
      
    \# Critical barrier: Forces CPU to wait for GPU queue to flush  
    torch.mps.synchronize()  
    end \= time.perf\_counter()  
      
    avg\_ms \= ((end \- start) / iterations) \* 1000  
    print(f"Average Execution Time: {avg\_ms:.2f} ms")  
    return avg\_ms

### **Snippet 9.4: CPU Fallback Registration Wrapper**

When deploying experimental OPs lacking MPS support, bind them safely to the CPU fallback to prevent runtime crashes during sequence compilation.

Python

import torch

\# 1\. Define the custom operation schema for the experimental kernel  
torch.library.define("mamba\_custom::esoteric\_scan", "(Tensor x, Tensor dt) \-\> Tensor")

def cpu\_esoteric\_scan(x: torch.Tensor, dt: torch.Tensor) \-\> torch.Tensor:  
    """Pure python/CPU implementation goes here"""  
    return x \* dt \# Placeholder mathematical logic

\# 2\. Register the CPU implementation  
torch.library.impl("mamba\_custom::esoteric\_scan", "cpu", cpu\_esoteric\_scan)

\# 3\. Force MPS to fallback to CPU for this specific operator  
def wrapper\_cpu\_fallback(op, \*args, \*\*kwargs):  
    """  
    Moves arguments to CPU, executes, and moves back to MPS.  
    WARNING: Overuse causes Objective-C autorelease pool memory leaks.  
    """  
    cpu\_args \=  
    result \= op(\*cpu\_args, \*\*kwargs)  
    return result.to('mps') if isinstance(result, torch.Tensor) else result

\# 4\. Bind using PrivateUse1 architecture rules for fallback routing  
@torch.library.impl("mamba\_custom::esoteric\_scan", "mps")  
def fallback\_esoteric\_scan\_mps(x: torch.Tensor, dt: torch.Tensor) \-\> torch.Tensor:  
    return wrapper\_cpu\_fallback(cpu\_esoteric\_scan, x, dt)

## **10\. Final "Red Flags" Checklist Before Shipping**

Before merging sequence-model code into the main branch or cutting a release for an edge deployment pipeline, systematically audit the following checklist. Failure to clear these flags will result in unpredictable behavior on target hardware.

1. **Contiguity Audit Cleared:** Are there any addcmul\_, addcdiv\_, or copy\_ operations executing on transposed or view-altered tensors without a preceding .contiguous() check? Ensure .is\_contiguous() assertions exist precisely at the boundaries of the Mamba module.3  
2. **Zero CUDA Artifacts:** Does running grep \-r "nvcc". or grep \-r "cuda". return any active dependencies in the build scripts? The library must compile purely through PyTorch C++ extensions or Python, or it will fail on macOS.1  
3. **Numerical Validation Passed:** Did the final Mamba-ASR-MPS model output match the CPU reference under strict FP16 tolerances (\< 1e-4) during the unit test phase?.19  
4. **FP32 Island Verification Verified:** Are the log, exp, and cumsum operations strictly casting their input tensors to torch.float32? Failure to enforce this guarantees exponential deterioration and NaN propagation on any continuous audio input exceeding 30 seconds.4  
5. **Memory Leak Check Stable:** Does torch.mps.current\_allocated\_memory() remain perfectly flat after the first 5 batches of the training loop? Unbounded upward growth indicates a retained computational graph, a missing .detach(), or a CPU fallback failing to release Apple Unified Memory properly.18  
6. **Dispatch Overhead Assessment Assessed:** For CNN-to-SSM pipelines, does running a batch size of 1 take virtually the identical amount of time as running a batch size of 8? If so, the operation is entirely dispatch-bound, and micro-kernels need to be fused or batched more aggressively before deployment.9  
7. **Frontend Stride Verification Confirmed:** Is the ASR CNN frontend compressing the raw 16kHz audio sufficiently? A raw waveform should be reduced to roughly 50Hz (20ms frames) prior to hitting the selective scan to maintain high execution performance and semantic accuracy.13

#### **Works cited**

1. mamba-ssm for macos with M1 · Issue \#324 · state-spaces/mamba, accessed April 2, 2026, [https://github.com/state-spaces/mamba/issues/324](https://github.com/state-spaces/mamba/issues/324)  
2. Xinguang/MiniMamba: A Minimal PyTorch Implementation ... \- GitHub, accessed April 2, 2026, [https://github.com/Xinguang/MiniMamba](https://github.com/Xinguang/MiniMamba)  
3. the bug that taught me more about PyTorch than years of using it \- Elana Simon, accessed April 2, 2026, [https://elanapearl.github.io/blog/2025/the-bug-that-taught-me-pytorch/](https://elanapearl.github.io/blog/2025/the-bug-that-taught-me-pytorch/)  
4. A Study on Energy Consumption in AI-Driven Medical Image Segmentation \- PMC, accessed April 2, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12194113/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12194113/)  
5. CUDA semantics \- PyTorch documentation, accessed April 2, 2026, [https://docs.pytorch.org/docs/stable/notes/cuda.html](https://docs.pytorch.org/docs/stable/notes/cuda.html)  
6. We compress any BF16 model to \~70% size during inference, while keeping the output LOSSLESS so that you can fit in more ERP context or run larger models. : r/LocalLLaMA \- Reddit, accessed April 2, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1k7o89n/we\_compress\_any\_bf16\_model\_to\_70\_size\_during/](https://www.reddit.com/r/LocalLLaMA/comments/1k7o89n/we_compress_any_bf16_model_to_70_size_during/)  
7. torch.mps.synchronize — PyTorch 2.11 documentation, accessed April 2, 2026, [https://docs.pytorch.org/docs/stable/generated/torch.mps.synchronize.html](https://docs.pytorch.org/docs/stable/generated/torch.mps.synchronize.html)  
8. Unleashing Apple Silicon's AI Power: A Deep Dive into MPS-Accelerated Image Generation, accessed April 2, 2026, [https://medium.com/@michael.hannecke/unleashing-apple-silicons-hidden-ai-superpower-a-technical-deep-dive-into-mps-accelerated-image-9573ba90570a](https://medium.com/@michael.hannecke/unleashing-apple-silicons-hidden-ai-superpower-a-technical-deep-dive-into-mps-accelerated-image-9573ba90570a)  
9. torch.roll runs too slow at MPS backend · Issue \#141789 \- GitHub, accessed April 2, 2026, [https://github.com/pytorch/pytorch/issues/141789](https://github.com/pytorch/pytorch/issues/141789)  
10. Speeding up PyTorch inference on Apple devices with AI-generated ..., accessed April 2, 2026, [https://gimletlabs.ai/blog/ai-generated-metal-kernels](https://gimletlabs.ai/blog/ai-generated-metal-kernels)  
11. Profiling Apple Silicon Performance for ML Training \- arXiv, accessed April 2, 2026, [https://arxiv.org/abs/2501.14925](https://arxiv.org/abs/2501.14925)  
12. Docker Model Runner Adds vLLM Support on macOS, accessed April 2, 2026, [https://www.docker.com/blog/docker-model-runner-vllm-metal-macos/](https://www.docker.com/blog/docker-model-runner-vllm-metal-macos/)  
13. HuBERT Features: Speech Representation Learning, accessed April 2, 2026, [https://www.emergentmind.com/topics/hubert-features](https://www.emergentmind.com/topics/hubert-features)  
14. Speech Separation Using Advanced Deep Neural Network Methods: A Recent Survey, accessed April 2, 2026, [https://www.mdpi.com/2504-2289/9/11/289](https://www.mdpi.com/2504-2289/9/11/289)  
15. An Exploration of Mamba for Speech Self-Supervised Models \- arXiv, accessed April 2, 2026, [https://arxiv.org/html/2506.12606v1](https://arxiv.org/html/2506.12606v1)  
16. DuplexMamba: Enhancing Real-time Speech Conversations with Duplex and Streaming Capabilities \- arXiv, accessed April 2, 2026, [https://arxiv.org/html/2502.11123v3](https://arxiv.org/html/2502.11123v3)  
17. Operator Registration — PyTorch 2.11 documentation, accessed April 2, 2026, [https://docs.pytorch.org/docs/stable/accelerator/operators.html](https://docs.pytorch.org/docs/stable/accelerator/operators.html)  
18. Weekly GitHub Report for Pytorch: May 26, 2025 \- June 02, 2025 (12:05:35) \- Buttondown, accessed April 2, 2026, [https://buttondown.com/weekly-project-news/archive/weekly-github-report-for-pytorch-may-26-2025-june-5528/](https://buttondown.com/weekly-project-news/archive/weekly-github-report-for-pytorch-may-26-2025-june-5528/)  
19. Compiler-First State Space Duality and Portable O⁢(1) Autoregressive Caching for Inference \- arXiv, accessed April 2, 2026, [https://arxiv.org/html/2603.09555v1](https://arxiv.org/html/2603.09555v1)  
20. MPS or MLX for Domestic AI? The Answer Will Surprise You | by Mike Koypish \- Medium, accessed April 2, 2026, [https://medium.com/@koypish/mps-or-mlx-for-domestic-ai-the-answer-will-surprise-you-df4b111de8a0](https://medium.com/@koypish/mps-or-mlx-for-domestic-ai-the-answer-will-surprise-you-df4b111de8a0)  
21. \[Performance\] PyTorch (MPS) is faster than MLX in backward of convolution layer \#1313, accessed April 2, 2026, [https://github.com/ml-explore/mlx/issues/1313](https://github.com/ml-explore/mlx/issues/1313)  
22. Mamba: Linear-Time Sequence Modeling with Selective State Spaces \- arXiv, accessed April 2, 2026, [https://arxiv.org/html/2312.00752v2](https://arxiv.org/html/2312.00752v2)  
23. Mamba for Dummies: Efficient Linear-Time LLMs Explained \- Michiel Horstman \- Medium, accessed April 2, 2026, [https://michielh.medium.com/mamba-for-dummies-linear-time-llms-explained-0d4b51efcf9f](https://michielh.medium.com/mamba-for-dummies-linear-time-llms-explained-0d4b51efcf9f)  
24. Mamba-2: Algorithms and Systems | Princeton Language and Intelligence, accessed April 2, 2026, [https://pli.princeton.edu/blog/2024/mamba-2-algorithms-and-systems](https://pli.princeton.edu/blog/2024/mamba-2-algorithms-and-systems)  
25. Frequently Asked Questions — PyTorch 2.11 documentation, accessed April 2, 2026, [https://docs.pytorch.org/docs/stable/user\_guide/torch\_compiler/torch.compiler\_faq.html](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_faq.html)  
26. Goodbye API Keys, Hello Local LLMs: How I Cut Costs by Running LLM Models on my M3 MacBook | by Luke Kerbs | Medium, accessed April 2, 2026, [https://medium.com/@lukekerbs/goodbye-api-keys-hello-local-llms-how-i-cut-costs-by-running-llm-models-on-my-m3-macbook-a3074e24fee5](https://medium.com/@lukekerbs/goodbye-api-keys-hello-local-llms-how-i-cut-costs-by-running-llm-models-on-my-m3-macbook-a3074e24fee5)  
27. Towards Effective and Efficient Open Speech Foundation Models \- ProQuest, accessed April 2, 2026, [https://search.proquest.com/openview/b289a6f11d480237025789bca753bdbb/1?pq-origsite=gscholar\&cbl=18750\&diss=y](https://search.proquest.com/openview/b289a6f11d480237025789bca753bdbb/1?pq-origsite=gscholar&cbl=18750&diss=y)  
28. Lightweight Methods and Models for Practical Visual Speech Recognition from Video Sequences, accessed April 2, 2026, [https://www.cse.uoi.gr/wp-content/uploads/publications/PD-2026-1.pdf](https://www.cse.uoi.gr/wp-content/uploads/publications/PD-2026-1.pdf)  
29. MPS slows down after sleep · Issue \#124056 \- GitHub, accessed April 2, 2026, [https://github.com/pytorch/pytorch/issues/124056](https://github.com/pytorch/pytorch/issues/124056)  
30. Ultimate guide to PyTorch library in Python \- Deepnote, accessed April 2, 2026, [https://deepnote.com/blog/ultimate-guide-to-pytorch-library-in-python](https://deepnote.com/blog/ultimate-guide-to-pytorch-library-in-python)  
31. Maxtimer97/pytorch\_mamba: A simple and efficient Mamba implementation in PyTorch and MLX. \- GitHub, accessed April 2, 2026, [https://github.com/Maxtimer97/pytorch\_mamba](https://github.com/Maxtimer97/pytorch_mamba)  
32. Deep learning approaches for adaptive audio processing and binary classification in digital health \- Universität Augsburg, accessed April 2, 2026, [https://opus.bibliothek.uni-augsburg.de/opus4/files/111049/Diss\_Liu.pdf](https://opus.bibliothek.uni-augsburg.de/opus4/files/111049/Diss_Liu.pdf)  
33. \[D\] M4 chips for training ML? (MPS) : r/MachineLearning \- Reddit, accessed April 2, 2026, [https://www.reddit.com/r/MachineLearning/comments/1gf46km/d\_m4\_chips\_for\_training\_ml\_mps/](https://www.reddit.com/r/MachineLearning/comments/1gf46km/d_m4_chips_for_training_ml_mps/)  
34. torch.mps — PyTorch 2.11 documentation, accessed April 2, 2026, [https://docs.pytorch.org/docs/stable/mps.html](https://docs.pytorch.org/docs/stable/mps.html)  
35. Automatic Mixed Precision package \- torch.amp — PyTorch 2.11 documentation, accessed April 2, 2026, [https://docs.pytorch.org/docs/stable/amp.html](https://docs.pytorch.org/docs/stable/amp.html)  
36. Troubleshooting GuardOnDataDependentSymNode Errors — PyTorch 2.11 documentation, accessed April 2, 2026, [https://docs.pytorch.org/docs/stable/user\_guide/torch\_compiler/compile/dynamic\_shapes\_troubleshooting\_guardon\_errors.html](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/dynamic_shapes_troubleshooting_guardon_errors.html)  
37. What's new \- Modular docs, accessed April 2, 2026, [https://docs.modular.com/max/changelog/](https://docs.modular.com/max/changelog/)  
38. matmul() using PyTorch's MPS backend is faster than Apple's MLX \- Kevin Martin Jose, accessed April 2, 2026, [https://kevinmartinjose.com/2025/04/21/matmul-using-pytorchs-mps-backend-is-faster-than-apples-mlx/](https://kevinmartinjose.com/2025/04/21/matmul-using-pytorchs-mps-backend-is-faster-than-apples-mlx/)  
39. Native LLM and MLLM Inference at Scale on Apple Silicon \- arXiv, accessed April 2, 2026, [https://arxiv.org/html/2601.19139v1](https://arxiv.org/html/2601.19139v1)  
40. purohit10saurabh/mamba-ssm-macos: (Unofficial) Mamba ... \- GitHub, accessed April 2, 2026, [https://github.com/purohit10saurabh/mamba-ssm-macos](https://github.com/purohit10saurabh/mamba-ssm-macos)  
41. The Zero-One Specialization Problem — PyTorch 2.11 documentation, accessed April 2, 2026, [https://docs.pytorch.org/docs/stable/user\_guide/torch\_compiler/compile/dynamic\_shapes\_zero\_one\_specialization.html](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/dynamic_shapes_zero_one_specialization.html)  
42. Tensor Views — PyTorch 2.11 documentation, accessed April 2, 2026, [https://docs.pytorch.org/docs/stable/tensor\_view.html](https://docs.pytorch.org/docs/stable/tensor_view.html)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAYCAYAAAC8/X7cAAAD6ElEQVR4XrVWW4hOURReh5kx4zaEMJI8yC0v7oa8eCEpNEi5jUtGCnkQkZkHJUXk8iIP7sSLKOXyIITcQpoybo3Q4GFEmsZtfGuv8++zb+f8/zz46jt772+tvc7a13OIUhG1pxlU2ovsCNnWGAU52cjokmES5HUwUZCzOJlPU0+DY+0JlrmiRt6wnsjxOgf0gBLDMHQCp4Izwf6JnDrUCvA+lPL06O3GCPA22MU1ZGEokriIsh6sA3cioW8or4B9TUcDPNh7YHVOMMbQHXwBfgSbIolbmpgV9oON4EvwNXhHqRJkD8j5dBDXAIyXbUG9CeUSsMjQB4EP4PkWZddE1thNys6hzOm3lqIGza8o2ygeqLOG00jePdYUI5mAT+AqraRgL9gKjg85QVlG8vJaR+cX8ApVOgm5OAsTTwzHeOwagengoaRpzcVy8B1YkhZ/NklgI4AH3uPs81AnKpX14HPtZb/YQPSUHyTbg+NU2nbaAc6zlCROR/AnOEcrBsrh9wFlC0mSAj+JbhD/omw2TahfQ3HBUnzgXNG5uL6QZACnHc+bYB9bsqI1gAeTZoKNJAEPc0N38PPAjEVt0PmQCcTnFckZSOD3XQ2u4QpMxcQHOlIz2i92LgMfKU9ndozmZZBX0cN1UgOIVlpqOAke6CWtRHppa7RmQQc5Aw43DLUwtcG6PW7zAT5g2DWMNPim+pI0BcXw4IPLiU2wRuxXOHH222DofDuxxgmkAp5PcnHiaJh5NXDeurwivP/nhmZNoPR14B/i69Rw4/ub9zVTXY9+3gojSXzek32HjyYZAJcWjO7DKNn/CSI6RdJ3AXgL7BXr2sFqEi0myaEkkRiypzlQhdvZqB8j2WYcxLQOFp2qUidP9v5aVbPDTqL4VouZAdWRtxuvmIfjJMnN50Ygj2rwdyRL6AI3k+q7mRuBvozz4ChXjJ05cb4Y9tnGGPaEcp78W+EgUv85zeBdyi1jghXgL9JfQYGTKG+rI64YgxPn2PIL4viguZRkAPwdcuAF5O8HD8IHXMfh+QwVvhJ5pneBDdCvopzixrInho6CNxy9B55viL8tEbVA/wFxW86cuKkz2Ejy15kPn+G/yRVNFIFjwEUkf6B8wwjMjM3BSL2K5Hor9eYsgLw+YYeBYGskt5dC2K1AOJ35L7Ee4lat+IP0kaaHwTfWSVcUtC+Qgt8lmoXHd3CAa2GYC+gjKJKx2pNJztEQy+xDd3DqBYP3+InCOhayRAq8uvybPsM1xPA7B2fLd6MUsQ7s7YpBT8rpaVaFiaR+pTN97ED+04evRykjN+Hoaf5mMy3U/4L/vuyJyAtzXqRp4R/CE5wAFv4SjgAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAXCAYAAADUUxW8AAABRUlEQVR4Xo1SsUoEQQyNgiKIooVgYWdpYyVY2tpaiIWVvVjY2N/Z21j4Ff6BYGUhfoFg71kKJ6KcL5tMNpPsLj4uM8l7eTPZY4horvwMOfdrhSxkRmBMlhRB6O0ThJkNFb+O/MqJBeGr8vyc3GL5QcwXXqU+s+W7SH6xzxAbTh2CmR8QryTmnTRdiyScoLzBfkdiPqjUgo7DlhEviDXECA0z9ByHHo/qiDHiTPMLkpvPm6odMI3K2EY8UttzioVvHklZ2ioYe498zzGHPDY13145pdB1Eckzks1CausWydhPUgaomV/RZa00WCIxv0WBwS/nCPGJU1aiCCwgprhgStzr/qZ9ZBMVvpG/Y181GxE/zQ8S/QsxQVyr5r/ZM4E0uGv/BzF0H9p5kjfkhpYJmhk6zZlR9Bl8WfHWbcLwGhFu801DzB+fPSdSyYWxkQAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAYCAYAAADzoH0MAAABf0lEQVR4Xo1SPS9FQRCdFY2ofCQ0r1KJQrxC/AyFUFEjKrXfIRGVvOK95Cm1Gv+AQiHx0HATEQpU5Dl7Z+/szuwSJzm7M2dmZ2bvXiIPV69sGFtcVoyhUEwVO10LMK0M8gHz6QpJ/4WedhnrLYwn2BVCj/DvwQHsB+yX4B44bjrpKkAH6xD7ospztEasn2djsi/ZN2AVXZV9BXcIZV50U2yWuEtPlDjCKJY3uN/Yp3Us2uvEBbaN7nHgOLYfpYBk1ENOciuwJ8ApsA0eg9fgVjjS5GvHcdIr2AFPoIB0Cv0O+yo4Ikc8zBVmSN3fpfdfCrGuKGpjbID+C++oSwWE6b7AVh5l/8hxlwXdXMpVxPFW6bAH3t89ixf04GwSv/9Zo0ewM0dcvZ92BsbAXfATvAAnm3AD/0QD8J34/h/k/3tXa54v5H9dLlK/gB3eINxWu2KbqFLFLSKtaM8bg+20mzn8W48Ik5EPmK5WSrvJmoaMXsYfPdMeEXmg2MckePcHic08X283R18AAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAYCAYAAAAcYhYyAAABpElEQVR4XoWUvS8FQRTFZ8RXPFTiIyT+CQnRqihEQ0QpIVHoJAqi06hIFCioFCJqFKJRaGmEaPwDCk+IKJ7f7sy7c3d29znJmTn3zJ07d2ffPmNiWCUKtTj/QyfldUGJsgMLUlOU+fqELlhRZgNkTk3VDvyBNcK17FIQMka+xrJNihgzFj+XLiFmBiE+hFXC5shXyJfzEOsZXsXdlm4ThOwBhhrhOnMvnITD9dU8XPVQ2gXzxl3qGfocZ4X4FW7neskLiQ+Mu9QN5VLAvMMm5Tlk9oeWnhB30ZnH6KSIh27BRm1Y02dyXaQ5b4wXwROcwqn4ceaIkyLjdYPlUeMKL+hUm75+W0UkLyKDffjJQos41uwxfiM6mWfZPMO8CE/QX8y75AzpAx7hdaqCew9vvHMLKwgOsVs4R8TtOrmV4Jd5SRyHVfwH8pJCE8q/NO7n4OELMfWnoxR2grGHsTss2eRz+EC7+whb9FNprR3JHoEvXm/CjlRLkiCULyg6aNz3lfxtTDuroBftlBzAhZq2otX/oRtsbHrIWtRFhLzTCNGBor34A0xUMp5oT0amAAAAAElFTkSuQmCC>