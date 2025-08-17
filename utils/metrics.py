from __future__ import annotations

from typing import List


def levenshtein(a: List[str], b: List[str]) -> int:
    """Compute Levenshtein edit distance between two token sequences."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[n][m]


def wer(ref_text: str, hyp_text: str) -> float:
    """Compute word error rate (WER) as edit distance over words."""
    ref_words = ref_text.strip().split()
    hyp_words = hyp_text.strip().split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    ed = levenshtein(ref_words, hyp_words)
    return ed / len(ref_words)
