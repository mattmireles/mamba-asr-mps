# RNNT Bench Summary (2025-08-18T11:27:19)

Steps: 60, Batch: 2

| impl | fps | align_p50 | align_p90 | backend_usage |
|---|---:|---:|---:|---|
| mps_native | 819.5 | 4206.0 | 4881.6 | {'ta': 0, 'warp': 0, 'naive': 0, 'ctc': 0, 'cpu_grad': 4, 'unknown': 0} |
| auto | 1130.4 | 4662.0 | 5472.0 | {'ta': 0, 'warp': 0, 'naive': 0, 'ctc': 0, 'cpu_grad': 4, 'unknown': 0} |
| cpu_grad | 1124.6 | 3870.0 | 4721.2 | {'ta': 0, 'warp': 0, 'naive': 0, 'ctc': 0, 'cpu_grad': 4, 'unknown': 0} |
| ctc | 1193.5 | 3620.0 | 4541.4 | {'ta': 0, 'warp': 0, 'naive': 0, 'ctc': 4, 'cpu_grad': 0, 'unknown': 0} |