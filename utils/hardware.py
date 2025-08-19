"""
Hardware utility helpers for adaptive data loading and performance tuning.

Provides:
- get_optimal_worker_count(): Heuristic to size DataLoader workers based on CPU cores

Notes:
- On Apple Silicon, efficiency cores still help for IO-bound work.
- Cap the worker count to avoid excessive process overhead.
"""
from __future__ import annotations

import os


def get_optimal_worker_count(max_cap: int = 16) -> int:
    """Return a sensible DataLoader worker count based on CPU cores.

    Heuristic:
    - Use all available logical cores (os.cpu_count()),
      capped at ``max_cap`` (default 16) to avoid oversubscription.
    - Fallback to 4 if core count cannot be determined.
    """
    cpu_count = os.cpu_count() or 4
    # Ensure positive and cap
    if cpu_count < 1:
        cpu_count = 4
    return min(cpu_count, max_cap)
