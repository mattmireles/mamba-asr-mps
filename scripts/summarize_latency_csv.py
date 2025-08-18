#!/usr/bin/env python3
"""
Summarize latency CSV produced by MambaASRRunner (chunk,latency_ms).
Outputs a markdown table with count, avg, p50, p90, p99 and saves it.
"""
from __future__ import annotations
import argparse
import csv
import os
import statistics
from typing import List

def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    k = (len(data) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(data) - 1)
    if f == c:
        return data[f]
    d0 = data[f] * (c - k)
    d1 = data[c] * (k - f)
    return d0 + d1

def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize latency CSV (chunk,latency_ms)")
    ap.add_argument("--csv", required=True, help="Path to latency CSV")
    ap.add_argument("--out", required=True, help="Output markdown path")
    args = ap.parse_args()

    lat: List[float] = []
    with open(args.csv, newline="") as f:
        r = csv.DictReader(f)
        # Accept either columns: chunk,latency_ms or unnamed
        if r.fieldnames and "latency_ms" in r.fieldnames:
            for row in r:
                try:
                    lat.append(float(row["latency_ms"]))
                except Exception:
                    pass
        else:
            f.seek(0)
            r2 = csv.reader(f)
            next(r2, None)
            for row in r2:
                if len(row) >= 2:
                    try:
                        lat.append(float(row[1]))
                    except Exception:
                        pass

    lat.sort()
    count = len(lat)
    avg = statistics.mean(lat) if lat else 0.0
    p50 = percentile(lat, 50)
    p90 = percentile(lat, 90)
    p99 = percentile(lat, 99)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("## Streaming latency summary\n\n")
        f.write("| metric | ms |\n|---|---:|\n")
        f.write(f"| count | {count} |\n")
        f.write(f"| avg | {avg:.3f} |\n")
        f.write(f"| p50 | {p50:.3f} |\n")
        f.write(f"| p90 | {p90:.3f} |\n")
        f.write(f"| p99 | {p99:.3f} |\n")
    print(f"Wrote {args.out} with {count} samples: avg={avg:.3f} p50={p50:.3f} p90={p90:.3f} p99={p99:.3f}")

if __name__ == "__main__":
    main()
