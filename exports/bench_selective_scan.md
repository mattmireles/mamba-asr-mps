# selective_scan Benchmark (2025-08-18T11:35:19)

Command: `/Users/mattmireles/.pyenv/versions/3.10.0/bin/python /Users/mattmireles/Documents/GitHub/whisper/whisper-fine-tuner-macos/Mamba-ASR-MPS/benchmarks/bench_selective_scan.py --bench-iters 5 --warmup-iters 3 --sequence-lengths 256,512,1024,2048,4096,8192`

```
Seq Len | Throughput (tokens/sec)
--------|-------------------------
256     | 16563.41
512     | 14047.54
1024    | 16348.22
2048    | 16615.99
4096    | 16073.75
8192    | 16026.13
-------------------------------
```