# Latency Overview (10s, warmup=2)

Models: kd, kd_real, kd_real_beam3, opt, opt2, pruned, pruned_real, pruned_real_beam3, qat, qat_real, qat_real_beam3, w8

## cpu

| model | c128 | c256 | c512 |
|---|---:|---:|---:|
| kd | 3.35 | 3.30 | 3.43 |
| kd_real | - | 3.51 | - |
| kd_real_beam3 | - | 3.38 | - |
| opt | 4.30 | 4.17 | 4.06 |
| opt2 | 4.22 | 4.11 | 4.12 |
| pruned | 3.43 | 3.37 | 3.25 |
| pruned_real | - | 3.20 | - |
| pruned_real_beam3 | - | 3.03 | - |
| qat | 3.09 | 3.65 | 3.32 |
| qat_real | - | 3.65 | - |
| qat_real_beam3 | - | 3.42 | - |
| w8 | 4.29 | 4.15 | 4.21 |

## all

| model | c128 | c256 | c512 |
|---|---:|---:|---:|
| kd | - | 23.84 | - |
| kd_real | - | - | - |
| kd_real_beam3 | - | - | - |
| opt | - | 19.68 | - |
| opt2 | - | 18.76 | - |
| pruned | - | 20.57 | - |
| pruned_real | - | - | - |
| pruned_real_beam3 | - | - | - |
| qat | - | 26.54 | - |
| qat_real | - | - | - |
| qat_real_beam3 | - | - | - |
| w8 | - | 18.56 | - |

## cpu-gpu

| model | c128 | c256 | c512 |
|---|---:|---:|---:|
| kd | - | 15.70 | - |
| kd_real | - | - | - |
| kd_real_beam3 | - | - | - |
| opt | - | 18.63 | - |
| opt2 | - | 19.68 | - |
| pruned | - | 22.73 | - |
| pruned_real | - | - | - |
| pruned_real_beam3 | - | - | - |
| qat | - | 21.48 | - |
| qat_real | - | - | - |
| qat_real_beam3 | - | - | - |
| w8 | - | 18.69 | - |

