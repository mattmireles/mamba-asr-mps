#!/usr/bin/env python3
import sys
from pathlib import Path

# Writes a 1024x29 CSV in log-space that maps id->(id % 29) with 0 for matching group and -20 otherwise.
# Row 0 reserved for blank -> group 0.

def main(out_path: str) -> None:
    V = 1024
    K = 29
    out = []
    for i in range(V):
        row = ["-20"] * K
        g = 0 if i == 0 else (i % K)
        row[g] = "0"
        out.append(",".join(row))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(out))

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "Mamba-ASR-MPS/exports/projection_1024x29.modmap.csv"
    main(out)
