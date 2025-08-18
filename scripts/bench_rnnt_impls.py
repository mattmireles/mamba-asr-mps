#!/usr/bin/env python3
import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
TRAIN = REPO / 'train_RNNT.py'
EXPORTS = REPO / 'exports'
EXPORTS.mkdir(parents=True, exist_ok=True)

CASES = [
    ('mps_native', ['--rnnt_impl', 'mps_native']),
    ('auto', ['--rnnt_impl', 'auto']),
    ('cpu_grad', ['--force_cpu_grad']),
    ('ctc', ['--rnnt_impl', 'ctc']),
]

STEPS = 60
BATCH = 2

def run_case(name, args):
    json_path = EXPORTS / f'bench_{name}_{STEPS}.summary.json'
    csv_path = EXPORTS / f'bench_{name}_{STEPS}.csv'
    cmd = [
        PY, str(TRAIN), '--sanity', '--epochs', '1', '--batch_size', str(BATCH),
        '--max_steps', str(STEPS), '--log_json', str(json_path), '--log_csv', str(csv_path),
    ] + args
    env = os.environ.copy()
    env.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    proc = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout, json_path


def main():
    results = {}
    for name, args in CASES:
        rc, out, json_path = run_case(name, args)
        summary = {}
        if json_path.exists():
            try:
                summary = json.loads(json_path.read_text())
            except Exception:
                summary = {}
        results[name] = {
            'rc': rc,
            'encoder_fps': summary.get('encoder_fps'),
            'align_p50': summary.get('align_p50'),
            'align_p90': summary.get('align_p90'),
            'backend_usage': summary.get('backend_usage'),
        }
    def _fmt(x):
        try:
            return f"{float(x):.1f}"
        except Exception:
            return "-"

    md = [
        f"# RNNT Bench Summary ({datetime.now().isoformat(timespec='seconds')})",
        '',
        f"Steps: {STEPS}, Batch: {BATCH}",
        '',
        '| impl | fps | align_p50 | align_p90 | backend_usage |',
        '|---|---:|---:|---:|---|',
    ]
    for name in CASES:
        impl = name[0]
        r = results.get(impl, {})
        md.append(f"| {impl} | {_fmt(r.get('encoder_fps'))} | {_fmt(r.get('align_p50'))} | {_fmt(r.get('align_p90'))} | {r.get('backend_usage')} |")
    out_path = EXPORTS / 'bench_rnnt_summary.md'
    out_path.write_text('\n'.join(md))
    print(f"wrote {out_path}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
