#!/usr/bin/env python3
"""
Generates a CPU-op remediation table from an Instruments Core ML Operations CSV export.

Workflow:
1) Open the `.trace` in Xcode Instruments → Core ML instrument → Operations view
2) File → Export Table… → Save as CSV (e.g., exports/CoreMLTraces/operations.csv)
3) Run:
   python Mamba-ASR-MPS/scripts/coreml_ops_remediation.py \
     --csv Mamba-ASR-MPS/exports/CoreMLTraces/operations.csv \
     --out Mamba-ASR-MPS/exports/CoreMLTraces/coreml_cpu_ops.md

The script parses the CSV, filters rows where Location contains "CPU",
groups by operation type/name column, aggregates counts and total duration,
and outputs a markdown table along with suggested remediation strategies.

Notes for future maintainers (LLM-first):
- Instruments column names are not fully standardized across versions. The script
  attempts to find likely column headers case-insensitively. Supported aliases include:
  Operation/Op/Type/Operator, Location/Processor, Duration/Time.
- Remediation suggestions are heuristics maintained in `REMEDIATION_SUGGESTIONS`.
  Extend this map as you discover new CPU-bound ops.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def _find_column(headers: List[str], candidates: List[str]) -> int | None:
    """Return index of the first header matching any candidate (case-insensitive)."""
    lowered = [h.lower().strip() for h in headers]
    for candidate in candidates:
        cand = candidate.lower()
        for idx, header in enumerate(lowered):
            if cand == header:
                return idx
        # allow substring match (e.g., "operation type")
        for idx, header in enumerate(lowered):
            if cand in header:
                return idx
    return None


REMEDIATION_SUGGESTIONS: Dict[str, str] = {
    # Keep keys lowercase for robust matching
    "gather": "Replace with slice + concat during export, or pre-index tensors to avoid dynamic gather",
    "scatter": "Re-express as masked add or segment reductions supported by MPSGraph",
    "topk": "Approximate with partial sort; or argsort + slice if supported",
    "group_norm": "Fold into adjacent conv/linear or replace with instance norm where acceptable",
    "groupnorm": "Fold into adjacent conv/linear or replace with instance norm where acceptable",
    "einsum": "Expand into explicit matmul/transpose/batch-mm sequences supported by MPS",
    "index_select": "Pre-materialize indices; reshape/slice then concat to avoid dynamic index ops",
    "index_add": "Use segmented reductions or batched scatter-add patterns supported on MPS",
    "where": "Prefer boolean masking fused into upstream op; avoid shape-changing conditionals",
    "unique": "Avoid at runtime; precompute offline or use hashing tricks upstream",
    "sort": "Use small-K top-k approximations; avoid full sort on large tensors",
    "argsort": "Approximate with top-k + partial ranking if feasible",
}


def generate_markdown(
    cpu_ops: List[Tuple[str, int, float]], total_duration_ms: float
) -> str:
    lines: List[str] = []
    lines.append("| Operation Type (CPU) | Count | Total ms | % of CPU ms | Remediation |")
    lines.append("|---|---:|---:|---:|---|")
    for op_name, count, dur_ms in sorted(cpu_ops, key=lambda x: -x[2]):
        percent = (dur_ms / total_duration_ms * 100.0) if total_duration_ms > 0 else 0.0
        key = op_name.lower().strip()
        remediation = REMEDIATION_SUGGESTIONS.get(key)
        if remediation is None:
            # try partial matches
            for known in REMEDIATION_SUGGESTIONS:
                if known in key:
                    remediation = REMEDIATION_SUGGESTIONS[known]
                    break
        remediation = remediation or "Review graph; replace with supported MPSGraph primitives"
        lines.append(
            f"| {op_name} | {count} | {dur_ms:.2f} | {percent:.1f}% | {remediation} |"
        )
    return "\n".join(lines) + "\n"


def parse_csv(path: str) -> Tuple[List[Tuple[str, int, float]], float]:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return [], 0.0
    headers = rows[0]

    op_idx = _find_column(headers, ["operation", "op", "type", "operator", "operation type"])  # type: ignore[arg-type]
    loc_idx = _find_column(headers, ["location", "processor"])  # type: ignore[arg-type]
    dur_idx = _find_column(headers, ["duration", "time", "ms"])  # type: ignore[arg-type]

    if op_idx is None or loc_idx is None:
        raise ValueError(
            f"Could not find required columns in CSV. Headers: {headers}. "
            "Expected at least Operation and Location columns."
        )

    # Instruments sometimes provides duration in "Duration" or separate start/end. If no duration,
    # we fall back to count only. Duration parsing tries to extract a float (ms) if present.
    def _parse_ms(cell: str) -> float:
        if cell is None:
            return 0.0
        s = cell.strip().lower()
        # try plain float
        try:
            return float(s)
        except Exception:
            pass
        # try formats like "123.45 ms" or "0.12 s"
        if s.endswith(" ms"):
            try:
                return float(s[:-3].strip())
            except Exception:
                return 0.0
        if s.endswith(" s"):
            try:
                return float(s[:-2].strip()) * 1000.0
            except Exception:
                return 0.0
        return 0.0

    agg_count: Dict[str, int] = defaultdict(int)
    agg_ms: Dict[str, float] = defaultdict(float)

    for row in rows[1:]:
        if not row or len(row) <= max(op_idx, loc_idx, dur_idx or 0):
            continue
        op_raw = row[op_idx].strip() if op_idx is not None else "Unknown"
        loc_raw = row[loc_idx].strip().lower() if loc_idx is not None else ""

        if "cpu" not in loc_raw:
            continue
        dur_ms = _parse_ms(row[dur_idx]) if dur_idx is not None else 0.0

        agg_count[op_raw] += 1
        agg_ms[op_raw] += dur_ms

    cpu_total_ms = sum(agg_ms.values())
    cpu_ops: List[Tuple[str, int, float]] = [
        (op_name, agg_count[op_name], agg_ms[op_name]) for op_name in agg_count.keys()
    ]
    return cpu_ops, cpu_total_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CPU-op remediation table from Instruments CSV")
    parser.add_argument("--csv", required=True, help="Path to Operations CSV exported from Instruments")
    parser.add_argument("--out", required=True, help="Output markdown file path")
    args = parser.parse_args()

    cpu_ops, total_ms = parse_csv(args.csv)
    md = ["## Core ML Operations on CPU (from Instruments export)", ""]
    if not cpu_ops:
        md.append("No CPU-bound operations found or CSV missing duration/location columns.")
    else:
        md.append(generate_markdown(cpu_ops, total_ms))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(md))

    print(f"Wrote remediation table to: {args.out}")


if __name__ == "__main__":
    main()
