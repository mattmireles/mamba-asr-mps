# RNNT Bench Summary (2025-08-18T16:38:20)

Steps: 60, Batch: 2

| impl | fps | align_p50 | align_p90 | backend_usage |
|---|---:|---:|---:|---|
| mps_native | 1787.8 | 3847.5 | 5364.0 | {'ta': 0, 'warp': 0, 'naive': 0, 'ctc': 0, 'cpu_grad': 4, 'unknown': 0} |
| auto | 1527.3 | 4710.0 | 5710.0 | {'ta': 0, 'warp': 0, 'naive': 0, 'ctc': 0, 'cpu_grad': 4, 'unknown': 0} |
| cpu_grad | 1626.2 | 3381.0 | 4512.2 | {'ta': 0, 'warp': 0, 'naive': 0, 'ctc': 0, 'cpu_grad': 4, 'unknown': 0} |
| ctc | 839.8 | 2514.5 | 4101.2 | {'ta': 0, 'warp': 0, 'naive': 0, 'ctc': 4, 'cpu_grad': 0, 'unknown': 0} |