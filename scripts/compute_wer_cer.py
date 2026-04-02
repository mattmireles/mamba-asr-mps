#!/usr/bin/env python3
"""
Compute CER/WER for real-audio transcripts.

Usage:
  python scripts/compute_wer_cer.py \
    --ref exports/reference_10s.txt \
    --out exports/CoreMLTraces/wer_cer_overview.md \
    --glob "exports/transcript_*_*.txt"

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


# =============================================================================
# Named Constants for Text Processing and Evaluation
# =============================================================================

class TextProcessingConstants:
    """Named constants for WER/CER computation and text evaluation.
    
    These constants define formatting, precision, and threshold parameters for
    evaluation workflows. They replace magic numbers to provide clear documentation
    of their purpose and enable easy tuning for different evaluation scenarios.
    """
    
    # MARK: Report Formatting Constants
    
    # Decimal precision for error rate reporting.
    # Three decimal places provide sufficient precision for CER/WER analysis
    # while maintaining readability in evaluation reports.
    ERROR_RATE_PRECISION = 3
    
    # Default CER threshold for pass/fail evaluation in CI/CD pipelines.
    # Based on empirical analysis of acceptable model performance levels.
    DEFAULT_CER_THRESHOLD = 0.6
    
    # Default WER threshold for pass/fail evaluation in CI/CD pipelines.
    # Typically more lenient than CER due to word-level granularity.
    DEFAULT_WER_THRESHOLD = 0.8
    
    # MARK: Text Normalization Constants
    
    # Character set for normalization: lowercase letters, digits, apostrophes.
    # Optimized for English speech recognition evaluation.
    ALLOWED_CHARACTERS = set(string.ascii_lowercase + string.digits + "'")
    
    # MARK: File Processing Constants
    
    # Default glob pattern for transcript file discovery.
    # Matches standard MambaASRRunner output file naming convention.
    DEFAULT_TRANSCRIPT_PATTERN = "exports/transcript_*_*.txt"


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
    allowed = TextProcessingConstants.ALLOWED_CHARACTERS
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
    """Main execution function for CER/WER computation and evaluation.
    
    This function orchestrates the complete evaluation workflow from command-line
    argument parsing through final report generation. It handles batch processing
    of transcript files with configurable error thresholds for CI/CD integration.
    
    Workflow Overview:
    1. Parse command-line arguments with validation
    2. Load and normalize reference text
    3. Discover transcript files via glob pattern
    4. Process each transcript with error rate calculation
    5. Generate markdown report with results table
    6. Apply threshold gating for CI/CD integration
    
    Command-Line Interface:
        --ref: Path to ground-truth reference text file (required)
        --glob: Pattern for transcript file discovery (default: transcript_*_*.txt)
        --out: Output markdown file path (required)
        --cer-threshold: CER threshold for failure detection (optional)
        --wer-threshold: WER threshold for failure detection (optional)
        --strict: Fail on missing/empty transcripts (flag)
        --cer-only: Focus on CER evaluation, WER as informational (flag)
        
    Cross-File Integration:
        - Called by: scripts/eval_batch.sh for automated evaluation
        - Called by: CI/CD pipelines for regression detection
        - Input from: MambaASRRunner transcript outputs
        - Output to: Markdown reports for documentation and tracking
        
    Error Handling and Exit Codes:
        - 0: Normal completion, all thresholds met
        - 2: Missing/empty transcripts with --strict enabled
        - 3: Error rates exceed specified thresholds
        - SystemExit: Missing reference file with setup instructions
        
    Output Format:
        Generates markdown table with columns:
        - model: Extracted from filename (transcript_MODEL_mode.txt)
        - mode: Decoding mode (greedy, beam3, etc.)
        - CER: Character Error Rate (3 decimal places)
        - WER: Word Error Rate (3 decimal places)
        
    Performance Characteristics:
        - File I/O: Sequential processing of transcript files
        - Memory usage: O(max_file_size) for text processing
        - Processing time: O(files * avg_length^2) due to edit distance
        - Suitable for evaluation batches up to hundreds of files
        
    Example Usage Patterns:
        # Basic evaluation
        python scripts/compute_wer_cer.py --ref ref.txt --out results.md
        
        # CI/CD with strict gating
        python scripts/compute_wer_cer.py --ref ref.txt --out results.md \
               --cer-threshold 0.3 --strict --cer-only
               
        # Custom transcript pattern
        python scripts/compute_wer_cer.py --ref ref.txt --out results.md \
               --glob "outputs/*/transcript_*.txt"
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="Path to ground-truth reference text")
    ap.add_argument("--glob", default=TextProcessingConstants.DEFAULT_TRANSCRIPT_PATTERN, help="Glob for transcript files")
    ap.add_argument("--out", required=True, help="Output markdown path")
    ap.add_argument("--cer-threshold", type=float, default=None, help="Fail if any CER exceeds this value")
    ap.add_argument("--wer-threshold", type=float, default=None, help="Fail if any WER exceeds this value (informational if --cer-only)")
    ap.add_argument("--strict", action="store_true", help="Fail if any transcript is missing/empty")
    ap.add_argument("--cer-only", action="store_true", help="Compute and gate on CER only; WER reported as info")
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
    any_missing = False
    any_over_threshold = False
    repo_root = Path(__file__).resolve().parents[1]
    for tpath in sorted(repo_root.glob(args.glob)):
        hyp_text_raw = extract_text(tpath)
        if args.strict and hyp_text_raw.strip() == "":
            any_missing = True
        hyp_text = normalize_text_for_eval(hyp_text_raw)
        # CER: compare characters with spaces removed
        cer_err = levenshtein(list(hyp_text.replace(' ', '')), ref_chars)
        cer_rate = cer_err / max(1, len(ref_chars))
        # WER (informational if --cer-only)
        wer_err = levenshtein(hyp_text.split(), ref_words)
        wer_rate = wer_err / max(1, len(ref_words))
        # derive model/mode from filename: transcript_<model>_<mode>.txt
        m = re.match(r"transcript_(.+)_(greedy|beam3)\.txt$", tpath.name)
        model = m.group(1) if m else tpath.stem
        mode = m.group(2) if m else "-"
        out_rows.append((model, mode, cer_rate, wer_rate))
        if args.cer_threshold is not None and cer_rate > args.cer_threshold:
            any_over_threshold = True
        if not args.cer_only and args.wer_threshold is not None and wer_rate > args.wer_threshold:
            any_over_threshold = True

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# WER/CER Overview\n\n")
        f.write(f"Reference: {ref_path}\n\n")
        f.write("| model | mode | CER | WER |\n|---|---|---:|---:|\n")
        for model, mode, cer_rate, wer_rate in out_rows:
            precision = TextProcessingConstants.ERROR_RATE_PRECISION
            f.write(f"| {model} | {mode} | {cer_rate:.{precision}f} | {wer_rate:.{precision}f} |\n")
    print(f"Wrote {out_path}")
    if args.strict and any_missing:
        raise SystemExit(2)
    if any_over_threshold:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
