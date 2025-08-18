#!/usr/bin/env python3
"""
Summarize MPS hardware intervals exported from Instruments (XML via `xcrun xctrace export`).

Input: exports/CoreMLTraces/mps_hw_intervals.xml
Output: prints basic stats and writes a markdown summary if --out is provided.

The XML format contains a <schema name="mps-hw-intervals"> table with rows for MPSGraph events.
We extract duration (microseconds), count, and simple percentiles.
"""
from __future__ import annotations

import argparse
import os
import statistics
import xml.etree.ElementTree as ET
from typing import List


def parse_durations_us(xml_path: str) -> List[int]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    durations: List[int] = []
    # Rows look like: <row><start-time ...>...</start-time><duration ...>259167</duration> ...
    for dur_node in root.findall(".//duration"):
        try:
            val = int(dur_node.text.strip())
            durations.append(val)
        except Exception:
            continue
    return durations


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
    parser = argparse.ArgumentParser(description="Summarize MPS hardware intervals (XML)")
    parser.add_argument("--xml", default="Mamba-ASR-MPS/exports/CoreMLTraces/mps_hw_intervals.xml")
    parser.add_argument("--out", default="", help="Optional markdown output path")
    args = parser.parse_args()

    durations_us = parse_durations_us(args.xml)
    durations_us.sort()

    if not durations_us:
        print("No durations found.")
        return

    dur_ms = [d / 1000.0 for d in durations_us]
    count = len(dur_ms)
    total_ms = sum(dur_ms)
    mean_ms = statistics.mean(dur_ms)
    p50 = percentile(dur_ms, 50)
    p90 = percentile(dur_ms, 90)
    p99 = percentile(dur_ms, 99)

    print(f"count={count} total_ms={total_ms:.2f} mean_ms={mean_ms:.3f} p50={p50:.3f} p90={p90:.3f} p99={p99:.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            f.write("## MPS intervals summary (from XML)\n\n")
            f.write("| metric | value |\n|---|---:|\n")
            f.write(f"| count | {count} |\n")
            f.write(f"| total_ms | {total_ms:.2f} |\n")
            f.write(f"| mean_ms | {mean_ms:.3f} |\n")
            f.write(f"| p50_ms | {p50:.3f} |\n")
            f.write(f"| p90_ms | {p90:.3f} |\n")
            f.write(f"| p99_ms | {p99:.3f} |\n")
        print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
