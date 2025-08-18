## **Project Mamba-ASR-MPS: V2 Strategic Plan**

**Objective:** Pivot from an ANE-centric strategy to a CPU-first, data-driven optimization plan. The goal is to build the fastest possible on-device ASR system by leveraging the proven strengths of Apple Silicon's components: the exceptionally fast CPU for model inference and the AMX coprocessors for accelerated decoding.

Technical Reference guide: 'README/Mamba-on-Apple-Silicon.md' 

### **Phase 1: Characterize the `selective_scan` Performance Bottleneck**

**Goal:** To gather empirical data on the performance of our current pure-PyTorch `selective_scan` implementation at various sequence lengths. This will determine if a custom Metal kernel is a necessary, high-priority task or a future optimization.

**Step 1.1: Create the Benchmark Script**

Create a new file at `Mamba-ASR-MPS/benchmarks/bench_selective_scan.py` with the following content. This script will isolate the `selective_scan_fn` and measure its throughput under controlled conditions.

```python
# Mamba-ASR-MPS/benchmarks/bench_selective_scan.py

import torch
import time
import argparse
import sys
import os

# Add project root to path to allow direct import of modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

try:
    from Mamba_ASR_MPS.modules.mamba.selective_scan_interface import selective_scan_fn
except ImportError as e:
    print("Error: Could not import selective_scan_fn.")
    print(f"PYTHONPATH is: {sys.path}")
    print(f"Original error: {e}")
    sys.exit(1)

def run_benchmark(d_model: int, d_state: int, sequence_lengths: list[int], batch_size: int, warmup_iters: int, bench_iters: int):
    """
    Measures the throughput of the selective_scan_fn at various sequence lengths.
    """
    if not torch.backends.mps.is_available():
        print("MPS not available. This benchmark is for Apple Silicon GPUs.")
        return

    device = torch.device("mps")
    print(f"Using device: {device}")
    print(f"Config: d_model={d_model}, d_state={d_state}, batch_size={batch_size}\n")

    results = {}

    for seq_len in sequence_lengths:
        print(f"--- Benchmarking sequence length: {seq_len} ---")

        # Create dummy inputs on the MPS device
        u = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32)
        delta = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32)
        A = torch.randn(d_model, d_state, device=device, dtype=torch.float32)
        B = torch.randn(batch_size, seq_len, d_state, device=device, dtype=torch.float32)
        C = torch.randn(batch_size, seq_len, d_state, device=device, dtype=torch.float32)
        D = torch.randn(d_model, device=device, dtype=torch.float32)
        z = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32)
        delta_bias = torch.randn(d_model, device=device, dtype=torch.float32)
        h_init = torch.zeros(batch_size, d_model, d_state, device=device, dtype=torch.float32)

        # Warm-up iterations
        print("Running warm-up iterations...")
        for _ in range(warmup_iters):
            _ = selective_scan_fn(u, delta, A, B, C, D, z, delta_bias, h_init, softplus=True)
        torch.mps.synchronize()

        # Benchmark iterations
        print("Running benchmark iterations...")
        start_time = time.perf_counter()
        for _ in range(bench_iters):
            _ = selective_scan_fn(u, delta, A, B, C, D, z, delta_bias, h_init, softplus=True)
        torch.mps.synchronize()
        end_time = time.perf_counter()

        total_time = end_time - start_time
        total_tokens = batch_size * seq_len * bench_iters
        throughput = total_tokens / total_time
        results[seq_len] = throughput

        print(f"Throughput: {throughput:.2f} tokens/sec")
        print("-" * (29 + len(str(seq_len))))

    print("\n--- Summary ---")
    print("Seq Len | Throughput (tokens/sec)")
    print("--------|-------------------------")
    for seq_len, throughput in results.items():
        print(f"{seq_len:<7} | {throughput:.2f}")
    print("-" * 31)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark selective_scan_fn on MPS.")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension.")
    parser.add_argument("--d-state", type=int, default=16, help="State dimension.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (use 1 to isolate sequence length effect).")
    parser.add_argument("--warmup-iters", type=int, default=5, help="Number of warm-up iterations.")
    parser.add_argument("--bench-iters", type=int, default=10, help="Number of benchmark iterations.")
    parser.add_argument("--sequence-lengths", type=str, default="256,512,1024,2048,4096,8192", help="Comma-separated list of sequence lengths to test.")

    args = parser.parse_args()
    sequence_lengths = [int(x) for x in args.sequence_lengths.split(',')]

    run_benchmark(args.d_model, args.d_state, sequence_lengths, args.batch_size, args.warmup_iters, args.bench_iters)
```

**Step 1.2: Execute the Benchmark**

Run the newly created script from the root of the `whisper-fine-tuner-macos` directory using the following command. The `PYTHONPATH` ensures that the script can correctly import the `selective_scan_fn` module.

```bash
PYTHONPATH=. python Mamba-ASR-MPS/benchmarks/bench_selective_scan.py
```

**Step 1.3: Analyze and Document the Results**

1.  Observe the output table.
    *   **If throughput remains relatively stable or scales near-linearly**, it confirms that for these sequence lengths, the naive scan is not a catastrophic bottleneck.
    *   **If throughput drops significantly as sequence length increases**, it proves that the sequential nature of the operation is becoming the dominant performance factor.
2.  Create a new section in `Mamba-ASR-MPS/README/implementation-plan-v2.md` titled "**`selective_scan` Scaling Benchmark Results**".
3.  Copy the summary table from the script's output into this new section.
4.  Add a brief conclusion based on the results, stating whether a custom Metal kernel is an immediate priority or a future optimization.

### **Phase 2: Remediate the RNNT Training Bottleneck**

**Goal:** To significantly improve training throughput by eliminating the constant data transfer between the MPS device and the CPU for the RNNT loss calculation.

**Step 2.1: Research and Feasibility Study**

This is an investigative step. The goal is to determine the most viable path to a native MPS implementation of the RNN-T loss.

1.  **Investigate `warp-transducer`**: Search for maintained forks or branches of `warp-transducer` that have added experimental support for non-CUDA backends, specifically CPU or MPS. This is the lowest-effort, highest-reward path if it exists.
2.  **Investigate Pure PyTorch Implementation**: Research academic papers and open-source implementations (e.g., in K2, SpeechBrain, or ESPnet) of the RNN-T forward-backward algorithm. The goal is to understand if the algorithm can be expressed using only PyTorch operations that are known to be efficient on the MPS backend. This is a complex but potentially rewarding path.
3.  **Evaluate CTC as a Fallback**: If the above paths prove to be too complex or time-consuming, the final option is to switch the primary training loss to Connectionist Temporal Classification (CTC). Our `train_RNNT.py` script already supports this via `--rnnt_impl ctc`. This would allow us to train the powerful Mamba encoder without the loss bottleneck, providing a strong baseline model while the native RNNT loss is developed in parallel.

**Step 2.2: Implement the Chosen Solution**

Based on the research, execute the chosen path. This will likely involve creating a new file, `Mamba-ASR-MPS/modules/rnnt_loss_mps.py`, and integrating it into `train_RNNT.py` under a new `--rnnt_impl mps_native` option.

**Step 2.3: Benchmark and Document**

Run a comparative benchmark between the old `--force_cpu_grad` path and the new native path. Measure the training throughput (tokens/sec) on a standardized run (e.g., 512 samples, 300 steps). Document the percentage improvement in the implementation plan and the training notes.

### **Phase 3: Core ML Graph Analysis**

**Goal:** To get definitive, layer-by-layer data on which parts of our exported Core ML model are running on the CPU instead of the ANE.

**Step 3.1: Capture an Instruments Trace**

Use the `xcrun xctrace` command-line tool to launch the Swift runner and record a Core ML performance trace.

```bash
# Ensure the trace directory exists
mkdir -p Mamba-ASR-MPS/exports/CoreMLTraces

# Define variables for clarity
RUNNER="Mamba-ASR-MPS/swift/MambaASRRunner/.build/release/MambaASRRunner"
MODEL_MLC="Mamba-ASR-MPS/exports/Compiled_fp16_w8/MambaASR_fp16_w8.mlmodelc"
MODEL_MLPACKAGE="Mamba-ASR-MPS/exports/MambaASR_fp16_w8.mlpackage"
WAV_FILE="Mamba-ASR-MPS/exports/tts_real_long_16k.wav"
TRACE_OUTPUT="Mamba-ASR-MPS/exports/CoreMLTraces/fp16_w8_analysis.trace"

# Run the trace
xcrun xctrace record --template 'Core ML' --time-limit 30s --output "$TRACE_OUTPUT" --launch "$RUNNER" -- \
--mlmodelc "$MODEL_MLC" \
--mlpackage "$MODEL_MLPACKAGE" \
--stream --duration 60 --warmup 2 \
--wav "$WAV_FILE" --compute all

echo "Trace saved to: $TRACE_OUTPUT"
```

**Step 3.2: Analyze the Trace in Xcode Instruments**

1.  Open the generated `.trace` file in Xcode Instruments.
2.  Select the **Core ML** instrument from the track list on the left.
3.  In the detail pane at the bottom, select the **Operations** view.
4.  Click on the "Location" column header to sort the operations by the processor they ran on.
5.  Create a list of every unique operation type where the "Location" is **CPU**.

**Step 3.3: Document and Plan Remediation**

1.  Create a new section in `implementation-plan-v2.md` titled "**Core ML Graph Analysis Results**".
2.  In this section, add a table with two columns: "Operation Type (on CPU)" and "Potential Remediation Strategy".
3.  Fill this table with the list of CPU-bound operations from your analysis. For each one, propose a high-level plan to fix it (e.g., "Operation: `gather`. Remediation: Replace with `slice` and `concat` operations in the PyTorch graph before export.").


## Implementation Progress (write your notes below)