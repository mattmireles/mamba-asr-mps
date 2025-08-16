"""
LibriSpeech preparation (minimal) for local experimentation.
Downloads (optional) and builds CSV manifests expected by train script.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torchaudio


SAMPLERATE = 16000


@dataclass
class LSRow:
    path: str
    duration: float
    text: str


def scan_directory_for_wavs_text(data_dir: Path) -> List[LSRow]:
    rows: List[LSRow] = []
    for wav_path in data_dir.rglob("*.flac"):
        try:
            info = torchaudio.info(str(wav_path))
            dur = float(info.num_frames) / float(info.sample_rate)
        except Exception:
            dur = 0.0
        txt_path = wav_path.with_suffix(".txt")
        text = ""
        if txt_path.exists():
            try:
                text = txt_path.read_text(encoding="utf-8").strip()
            except Exception:
                text = ""
        rows.append(LSRow(str(wav_path), dur, text))
    return rows


def write_manifest(rows: List[LSRow], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "duration", "text"])  # header
        for r in rows:
            w.writerow([r.path, f"{r.duration:.3f}", r.text])


def prepare_librispeech(root: str, split: str = "train-clean-100") -> Path:
    """
    Build a CSV manifest for a LibriSpeech-like directory.
    Expects data under: {root}/{split}
    """
    data_dir = Path(root) / split
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")
    rows = scan_directory_for_wavs_text(data_dir)
    out_csv = Path(root) / f"{split}.csv"
    write_manifest(rows, out_csv)
    return out_csv


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--split", type=str, default="train-clean-100")
    args = ap.parse_args()

    path = prepare_librispeech(args.root, args.split)
    print(f"Wrote manifest: {path}")
