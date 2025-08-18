#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
SCRIPT = REPO / 'benchmarks' / 'bench_selective_scan.py'
OUT = REPO / 'exports' / 'bench_selective_scan.md'

SEQS = '256,512,1024,2048,4096,8192'

cmd = [PY, str(SCRIPT), '--bench-iters', '5', '--warmup-iters', '3', '--sequence-lengths', SEQS]
proc = subprocess.run(cmd, cwd=str(REPO), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
text = proc.stdout
# Extract summary lines after '--- Summary ---'
lines = text.splitlines()
summary_idx = next((i for i,l in enumerate(lines) if l.strip().startswith('--- Summary ---')), None)
summary_block = '\n'.join(lines[summary_idx+1:]) if summary_idx is not None else text

md = []
md.append(f"# selective_scan Benchmark ({datetime.now().isoformat(timespec='seconds')})")
md.append('')
md.append(f"Command: `{ ' '.join(cmd) }`")
md.append('')
md.append('```')
md.append(summary_block.strip())
md.append('```')
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text('\n'.join(md))
print(f"wrote {OUT}")
