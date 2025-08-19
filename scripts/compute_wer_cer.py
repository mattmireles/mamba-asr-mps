#!/usr/bin/env python3
"""
Compute CER/WER for real-audio transcripts.

Usage:
  python scripts/compute_wer_cer.py \
    --ref Mamba-ASR-MPS/exports/reference_10s.txt \
    --out Mamba-ASR-MPS/exports/CoreMLTraces/wer_cer_overview.md \
    --glob "Mamba-ASR-MPS/exports/transcript_*_*.txt"

Notes:
- Expects transcripts captured from MambaASRRunner stdout redirection
- If reference not found, exits with informative message
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
from typing import List, Tuple
import string


def levenshtein(a: List[str], b: List[str]) -> int:
    la, lb = len(a), len(b)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[la][lb]


def normalize_text_for_eval(text: str) -> str:
    """Lowercase and remove non-linguistic punctuation while keeping apostrophes.

    This normalization aligns reference and hypothesis for fair WER/CER.
    - Keep: a-z, digits (for robustness), apostrophes
    - Map everything else to spaces, then collapse spaces
    """
    text = text.lower()
    allowed = set(string.ascii_lowercase + string.digits + "'")
    mapped_chars: List[str] = []
    for ch in text:
        mapped_chars.append(ch if ch in allowed else ' ')
    norm = ''.join(mapped_chars)
    norm = ' '.join(norm.split())  # collapse whitespace
    return norm.strip()


def extract_text(transcript_path: Path) -> str:
    """Extract the final decoded transcript line from runner output.

    Preference order:
    1) A line starting with the note icon and containing "Greedy transcript:" or "Beam transcript:"
    2) Any line that contains "transcript:" but NOT "(ids)" (to avoid raw id dumps)
    Returns empty string if nothing reasonable is found.
    """
    txt = transcript_path.read_text(errors="ignore")
    lines = txt.splitlines()
    greedy_or_beam: List[str] = []
    generic_transcripts: List[str] = []
    for line in lines:
        lowered = line.lower()
        if 'transcript:' in lowered and '(ids)' not in lowered:
            # capture after the first colon
            try:
                content = line.split(':', 1)[1].strip()
            except Exception:
                content = ''
            # prioritize explicit greedy/beam labels
            if 'greedy transcript' in lowered or 'beam transcript' in lowered:
                greedy_or_beam.append(content)
            else:
                generic_transcripts.append(content)

    if greedy_or_beam:
        return greedy_or_beam[-1]
    if generic_transcripts:
        return generic_transcripts[-1]
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="Path to ground-truth reference text")
    ap.add_argument("--glob", default="Mamba-ASR-MPS/exports/transcript_*_*.txt", help="Glob for transcript files")
    ap.add_argument("--out", required=True, help="Output markdown path")
    args = ap.parse_args()

    ref_path = Path(args.ref)
    if not ref_path.exists():
        raise SystemExit(f"Reference file not found: {ref_path}\nPlace the ground-truth text here and re-run.")
    ref_text_raw = ref_path.read_text().strip()
    ref_text = normalize_text_for_eval(ref_text_raw)
    # CER: character-level, ignore spaces by removing them after normalization
    ref_chars = list(ref_text.replace(' ', ''))
    # WER: word-level on normalized text
    ref_words = ref_text.split()

    out_rows: List[Tuple[str, str, float, float]] = []
    for tpath in sorted(Path().glob(args.glob)):
        hyp_text_raw = extract_text(tpath)
        hyp_text = normalize_text_for_eval(hyp_text_raw)
        # CER: compare characters with spaces removed
        cer_err = levenshtein(list(hyp_text.replace(' ', '')), ref_chars)
        cer_rate = cer_err / max(1, len(ref_chars))
        # WER
        wer_err = levenshtein(hyp_text.split(), ref_words)
        wer_rate = wer_err / max(1, len(ref_words))
        # derive model/mode from filename: transcript_<model>_<mode>.txt
        m = re.match(r"transcript_(.+)_(greedy|beam3)\.txt$", tpath.name)
        model = m.group(1) if m else tpath.stem
        mode = m.group(2) if m else "-"
        out_rows.append((model, mode, cer_rate, wer_rate))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# WER/CER Overview\n\n")
        f.write(f"Reference: {ref_path}\n\n")
        f.write("| model | mode | CER | WER |\n|---|---|---:|---:|\n")
        for model, mode, cer_rate, wer_rate in out_rows:
            f.write(f"| {model} | {mode} | {cer_rate:.3f} | {wer_rate:.3f} |\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
