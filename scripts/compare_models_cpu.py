#!/usr/bin/env python3
from pathlib import Path
import re

def parse_cpu_avg(md_path: Path):
    if not md_path.exists():
        return None
    text = md_path.read_text()
    # Find cpu_c256 section and avg value
    mode = None
    avg = None
    for line in text.splitlines():
        m = re.match(r"##\s*(cpu_c256)\s*$", line.strip())
        if m:
            mode = m.group(1)
            continue
        if mode == 'cpu_c256' and line.strip().startswith('| avg'):
            parts = [p.strip() for p in line.strip().strip('|').split('|')]
            if len(parts) >= 2:
                try:
                    avg = float(parts[1])
                except Exception:
                    pass
            break
    return avg


def main():
    # Prefer fresh per-run CSVs/tables if present in training notes/plan; fallback to older sweep markdowns
    plan = Path("README/implementation-plan-v2.md").read_text()
    def extract_from_plan(model: str):
        # Look for CPU-only table values
        for line in plan.splitlines():
            if line.strip().startswith(f"| {model} |"):
                try:
                    return float(line.split('|')[2])
                except Exception:
                    return None
        return None
    models = ['base','opt','opt2','w8']
    rows = []
    for name in models:
        avg = extract_from_plan(name)
        rows.append((name, avg))
    print('# CPU latency (chunk=256) comparison')
    print()
    print('| model | cpu avg ms |')
    print('|---|---:|')
    for name, avg in rows:
        val = '-' if avg is None else f"{avg:.3f}"
        print(f"| {name} | {val} |")

if __name__ == '__main__':
    main()
