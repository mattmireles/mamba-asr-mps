# Mamba-ASR-MPS/benchmarks/bench_selective_scan.py

import torch
import time
import argparse
import sys
import os

# Add repo root and Mamba-ASR-MPS to path to allow direct import of modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

mamba_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, mamba_pkg_root)

# Resolve selective_scan function
selective_scan_fn = None
try:
    # Prefer package-local import when running from repo root
    from modules.mamba.selective_scan_interface import selective_scan as selective_scan_fn  # type: ignore
except Exception as e1:
    try:
        # Fallback to the name used in the implementation plan (underscored variant)
        from Mamba_ASR_MPS.modules.mamba.selective_scan_interface import selective_scan as selective_scan_fn  # type: ignore
    except Exception as e2:
        print("Error: Could not import selective_scan function.")
        print(f"sys.path is: {sys.path}")
        print(f"Primary import error: {e1}")
        print(f"Secondary import error: {e2}")
        sys.exit(1)


def run_benchmark(d_model: int, d_state: int, sequence_lengths: list[int], batch_size: int, warmup_iters: int, bench_iters: int) -> dict[int, float]:
    """
    Measures the throughput of the selective_scan_fn at various sequence lengths.
    Returns a mapping of seq_len -> tokens/sec.
    """
    if not torch.backends.mps.is_available():
        print("MPS not available. This benchmark is for Apple Silicon GPUs.")
        return {}

    device = torch.device("mps")
    print(f"Using device: {device}")
    print(f"Config: d_model={d_model}, d_state={d_state}, batch_size={batch_size}\n")

    results: dict[int, float] = {}

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
            _ = selective_scan_fn(u, delta, A, B, C, D, z, delta_bias, h_init)
        torch.mps.synchronize()

        # Benchmark iterations
        print("Running benchmark iterations...")
        start_time = time.perf_counter()
        for _ in range(bench_iters):
            _ = selective_scan_fn(u, delta, A, B, C, D, z, delta_bias, h_init)
        torch.mps.synchronize()
        end_time = time.perf_counter()

        total_time = end_time - start_time
        total_tokens = batch_size * seq_len * bench_iters
        throughput = total_tokens / total_time if total_time > 0 else float('inf')
        results[seq_len] = throughput

        print(f"Throughput: {throughput:.2f} tokens/sec")
        print("-" * (29 + len(str(seq_len))))

    print("\n--- Summary ---")
    print("Seq Len | Throughput (tokens/sec)")
    print("--------|-------------------------")
    for seq_len in sequence_lengths:
        tput = results.get(seq_len, float('nan'))
        print(f"{seq_len:<7} | {tput:.2f}")
    print("-" * 31)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark selective_scan_fn on MPS.")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension.")
    parser.add_argument("--d-state", type=int, default=16, help="State dimension.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (use 1 to isolate sequence length effect).")
    parser.add_argument("--warmup-iters", type=int, default=5, help="Number of warm-up iterations.")
    parser.add_argument("--bench-iters", type=int, default=10, help="Number of benchmark iterations.")
    parser.add_argument("--sequence-lengths", type=str, default="256,512,1024,2048,4096,8192", help="Comma-separated list of sequence lengths to test.")

    args = parser.parse_args()
    sequence_lengths = [int(x) for x in args.sequence_lengths.split(',') if x]

    run_benchmark(args.d_model, args.d_state, sequence_lengths, args.batch_size, args.warmup_iters, args.bench_iters)
