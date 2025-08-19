#!/usr/bin/env python3
import argparse
from pathlib import Path
import re

def parse_summary(md: str):
    # Return dict: {mode: {metric: value}}
    out = {}
    current = None
    for line in md.splitlines():
        m = re.match(r"##\s+([\w-]+)\s*$", line.strip())
        if m:
            current = m.group(1)
            out[current] = {}
            continue
        if '|' in line and line.strip().startswith('|') and 'metric' not in line:
            parts = [p.strip() for p in line.strip().strip('|').split('|')]
            if len(parts) == 2 and current:
                key, val = parts
                try:
                    out[current][key] = float(val)
                except Exception:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='Path to baseline sweep md')
    ap.add_argument('--new', required=True, help='Path to new sweep md')
    args = ap.parse_args()

    base = Path(args.base).read_text()
    new = Path(args.new).read_text()
    b = parse_summary(base)
    n = parse_summary(new)

    modes = sorted(set(b.keys()) | set(n.keys()))
    lines = ["# Latency sweep delta (new - base)",""]
    for mode in modes:
        lines.append(f"## {mode}")
        lines.append("")
        bb = b.get(mode, {})
        nn = n.get(mode, {})
        for metric in ['avg','p50','p90','p99']:
            if metric in bb and metric in nn:
                delta = nn[metric] - bb[metric]
                lines.append(f"- {metric}: {nn[metric]:.3f} (Δ {delta:+.3f}) vs {bb[metric]:.3f}")
        lines.append("")
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
