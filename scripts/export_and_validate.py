#!/usr/bin/env python3
"""
End-to-end export + compile + Swift validation runner for Core ML models.

Usage example:
  python scripts/export_and_validate.py \
    --checkpoint Mamba-ASR-MPS/checkpoints/kd_student.pt \
    --name MambaASR_kd --duration 10

Optional:
  --wav /path/to/16k_mono.wav  # stream real audio
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Ensure repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
MPS_ROOT = REPO_ROOT / "Mamba-ASR-MPS"
SCRIPTS_DIR = MPS_ROOT / "scripts"
RUNNER_DIR = MPS_ROOT / "swift" / "MambaASRRunner"
RUNNER_BIN = RUNNER_DIR / ".build" / "release" / "MambaASRRunner"


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None):
    print("$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint to export")
    ap.add_argument("--name", default="MambaASR_export", help="Base name for exported artifacts")
    ap.add_argument("--duration", type=int, default=5, help="Streaming duration (seconds)")
    ap.add_argument("--warmup", type=int, default=2, help="Number of warmup inferences to amortize first-call cost")
    ap.add_argument("--wav", type=str, default="", help="Optional 16kHz mono wav to stream")
    ap.add_argument("--latency_csv", type=str, default="", help="Optional path to write per-chunk latency CSV")
    ap.add_argument("--vocab_out", type=str, default="", help="Optional path to write vocab JSON for Swift greedy decode")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}")
        sys.exit(1)

    exports = MPS_ROOT / "exports"
    compiled_dir = exports / f"Compiled_{args.name}"
    mlpackage = exports / f"{args.name}.mlpackage"
    mlmodelc = compiled_dir / f"{args.name}.mlmodelc"
    exports.mkdir(parents=True, exist_ok=True)

    # 1) Export to .mlpackage
    env = os.environ.copy()
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # Ensure PYTHONPATH so export script can import modules
    env["PYTHONPATH"] = str(MPS_ROOT)
    run([sys.executable, str(SCRIPTS_DIR / "export_coreml.py"),
         "--model", str(ckpt), "--output", str(mlpackage)], env=env)

    # 2) Compile to .mlmodelc
    compiled_dir.mkdir(parents=True, exist_ok=True)
    run([
        "xcrun", "coremlcompiler", "compile",
        str(mlpackage), str(compiled_dir)
    ])

    # 3) Build Swift runner if missing
    if not RUNNER_BIN.exists():
        run(["swift", "build", "-c", "release", "--package-path", str(RUNNER_DIR)])

    # 3.5) Optionally emit a simple character vocab JSON for greedy decode
    vocab_path: Path | None = None
    if args.vocab_out:
        vocab_path = Path(args.vocab_out).resolve()
    else:
        vocab_path = (exports / "vocab_char_29.json").resolve()
    try:
        import json
        vocab_map: dict[str, str] = {"0": ""}
        vocab_map["1"] = " "
        for i, ch in enumerate([chr(ord('a') + k) for k in range(26)], start=2):
            vocab_map[str(i)] = ch
        vocab_map["28"] = "'"
        with open(vocab_path, "w") as f:
            json.dump(vocab_map, f)
    except Exception:
        vocab_path = None

    # 4) Run Swift runner in streaming mode
    cmd = [
        str(RUNNER_BIN),
        "--mlmodelc", str(mlmodelc),
        "--mlpackage", str(mlpackage),
        "--stream", "--duration", str(args.duration),
        "--warmup", str(args.warmup)
    ]
    if args.wav:
        cmd.extend(["--wav", args.wav])
    if args.latency_csv:
        cmd.extend(["--latency-csv", args.latency_csv])
    if vocab_path is not None:
        cmd.extend(["--vocab", str(vocab_path)])
    run(cmd)

    print("All done.")


if __name__ == "__main__":
    main()
