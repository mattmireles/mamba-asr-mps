#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

def parse(md_text: str):
    sections = {}
    current = None
    for line in md_text.splitlines():
        m = re.match(r"##\s*(cpu_c(\d+))\s*$", line.strip())
        if m:
            current = m.group(1)
            sections[current] = {}
            continue
        if current and line.strip().startswith('|') and 'metric' not in line:
            parts = [p.strip() for p in line.strip().strip('|').split('|')]
            if len(parts) >= 2:
                key, val = parts[0], parts[1]
                try:
                    sections[current][key] = float(val)
                except Exception:
                    pass
    rows = []
    for key in sorted(sections.keys(), key=lambda k: int(k.split('c')[-1])):
        chunk = int(key.split('c')[-1])
        s = sections[key]
        rows.append((chunk, s.get('avg'), s.get('p50'), s.get('p90'), s.get('p99')))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--md', required=True, help='Path to sweep markdown with cpu_c* sections')
    args = ap.parse_args()
    text = Path(args.md).read_text()
    rows = parse(text)
    print('# CPU chunk-size sweep summary')
    print()
    print('| chunk | avg ms | p50 | p90 | p99 |')
    print('|---:|---:|---:|---:|---:|')
    for chunk, avg, p50, p90, p99 in rows:
        def f(x):
            return '-' if x is None else f"{x:.3f}"
        print(f"| {chunk} | {f(avg)} | {f(p50)} | {f(p90)} | {f(p99)} |")

if __name__ == '__main__':
    main()
