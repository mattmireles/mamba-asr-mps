#!/usr/bin/env python3
import csv
import os
import sys
from statistics import mean
from typing import List


def summarize_csv(path: str) -> str:
    try:
        with open(path, newline='') as fh:
            reader = csv.DictReader(fh)
            losses = []
            aligns = []
            t_caps = []
            u_caps = []
            backends = []
            for row in reader:
                try:
                    if 'loss' in row and row['loss']:
                        losses.append(float(row['loss']))
                    if 'align' in row and row['align']:
                        aligns.append(int(row['align']))
                    if "t_cap" in row and row['t_cap']:
                        t_caps.append(int(row['t_cap']))
                    if "u_cap" in row and row['u_cap']:
                        u_caps.append(int(row['u_cap']))
                    if 'backend' in row and row['backend']:
                        backends.append(row['backend'])
                except Exception:
                    continue
        n = len(losses)
        if n == 0:
            return f"{os.path.basename(path)}: no rows"
        backend_counts = {}
        for b in backends:
            backend_counts[b] = backend_counts.get(b, 0) + 1
        parts = [
            f"{os.path.basename(path)}:",
            f"n={n}",
            f"mean_loss={mean(losses):.3f}",
            f"mean_align={mean(aligns):.1f}" if aligns else "mean_align=nan",
            f"mean_Tcap={mean(t_caps):.1f}" if t_caps else "mean_Tcap=nan",
            f"mean_Ucap={mean(u_caps):.1f}" if u_caps else "mean_Ucap=nan",
            f"backend_mix={backend_counts}" if backend_counts else "backend_mix={}"
        ]
        return " ".join(parts)
    except FileNotFoundError:
        return f"{os.path.basename(path)}: not found"


def main(argv: List[str]) -> int:
    base = os.path.join(os.path.dirname(__file__), '..', 'exports')
    base = os.path.abspath(base)
    if len(argv) > 1:
        files = argv[1:]
    else:
        files = [
            'rnnt_mps_native_60.csv',
            'rnnt_auto_60.csv',
            'rnnt_cpu_grad_60.csv',
            'rnnt_mps_native_60_align40k.csv',
            'rnnt_ctc_60.csv'
        ]
    for f in files:
        p = f if os.path.isabs(f) else os.path.join(base, f)
        print(summarize_csv(p))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
