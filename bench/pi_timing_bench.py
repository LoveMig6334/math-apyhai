#!/usr/bin/env python3
"""
Fixed-workload single-core Chudnovsky π timing benchmark.

Unlike ``singel_pi_bench.py`` (which fits a time budget and reports how many
terms fit), this script pins the workload at a *fixed* 500,000 Chudnovsky
terms and reports the wall-clock time. The time itself is the score: lower is
faster. Because the same number of bit-operations runs on every machine, the
results are directly comparable across CPUs.

The work is pure arbitrary-precision integer arithmetic via binary splitting.
It holds the GIL and is naturally single-threaded, so this measures exactly one
core. On Apple Silicon, macOS schedules it on a performance core automatically.

Usage:
    python3 pi_timing_bench.py                 # 500k terms, best of 3 runs
    python3 pi_timing_bench.py --terms 250000  # override the term count
    python3 pi_timing_bench.py --runs 5        # more runs, report the best
    python3 pi_timing_bench.py --full          # also time the final sqrt+divide
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from math import isqrt

sys.setrecursionlimit(10_000)

# Chudnovsky constants
A = 13_591_409
B = 545_140_134
C = 640_320
C3_OVER_24 = C**3 // 24
DIGITS_PER_TERM = 14.1816474627254776555

DEFAULT_TERMS = 500_000
DEFAULT_RUNS = 3


def bs(a: int, b: int) -> tuple[int, int, int]:
    """Binary splitting: returns (P, Q, T) for the partial sum on [a, b)."""
    if b - a == 1:
        if a == 0:
            P = Q = 1
        else:
            P = (6 * a - 5) * (2 * a - 1) * (6 * a - 1)
            Q = a * a * a * C3_OVER_24
        T = P * (A + B * a)
        if a & 1:
            T = -T
        return P, Q, T
    m = (a + b) // 2
    Pl, Ql, Tl = bs(a, m)
    Pr, Qr, Tr = bs(m, b)
    return Pl * Pr, Ql * Qr, Qr * Tl + Pl * Tr


def time_terms(n: int) -> float:
    """Time just the binary-splitting summation of N Chudnovsky terms.

    This is the 'calculate 500,000 terms' workload proper — the series sum,
    excluding the one-off final square root and division."""
    t0 = time.perf_counter()
    bs(0, n)
    return time.perf_counter() - t0


def time_full(n: int) -> float:
    """Time the complete π pipeline: series sum + final isqrt + divide."""
    digits = int(n * DIGITS_PER_TERM)
    t0 = time.perf_counter()
    _, Q, T = bs(0, n)
    sqrt_c = isqrt(10005 * 10 ** (2 * digits))
    _ = (Q * 426880 * sqrt_c) // T
    return time.perf_counter() - t0


def fmt(seconds: float) -> str:
    """Human-friendly duration: '12.345 s' or '1 m 02.345 s'."""
    if seconds < 60:
        return f"{seconds:.3f} s"
    m, s = divmod(seconds, 60)
    return f"{int(m)} m {s:06.3f} s"


def machine_fingerprint() -> dict[str, str]:
    """Identifying details so results from different machines are comparable."""
    uname = platform.uname()
    return {
        "machine": f"{uname.system} {uname.release} ({uname.machine})",
        "processor": platform.processor() or uname.processor or "unknown",
        "python": f"{platform.python_implementation()} {platform.python_version()}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--terms",
        type=int,
        default=DEFAULT_TERMS,
        help=f"number of Chudnovsky terms (default: {DEFAULT_TERMS:,})",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"timed runs; the best (fastest) is the score (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="time the full pipeline (sum + sqrt + divide) instead of just the sum",
    )
    args = parser.parse_args()

    if args.terms < 1 or args.runs < 1:
        parser.error("--terms and --runs must both be >= 1")

    n = args.terms
    digits = int(n * DIGITS_PER_TERM)
    measure = time_full if args.full else time_terms
    workload = "full pipeline (sum + sqrt + divide)" if args.full else "series sum"
    info = machine_fingerprint()

    print("=" * 64)
    print("  Fixed-workload single-core Chudnovsky π timing")
    print("=" * 64)
    print(f"  Machine     : {info['machine']}")
    print(f"  Processor   : {info['processor']}")
    print(f"  Runtime     : {info['python']}")
    print(f"  Workload    : {n:,} terms  ({workload})")
    print(f"  Equivalent  : ~{digits:,} decimal digits of π")
    print("=" * 64)

    print(f"\nTiming {args.runs} run(s) (best is the score)...")
    print(f"  {'run':>4}  {'time':>14}")

    best = float("inf")
    for i in range(1, args.runs + 1):
        t = measure(n)
        best = min(best, t)
        flag = "  <- best" if t == best else ""
        print(f"  {i:>4}  {fmt(t):>14}{flag}")

    print()
    print("=" * 64)
    print(f"  SCORE (best of {args.runs}): {fmt(best)}")
    print("=" * 64)
    print(f"  Throughput  ≈ {n / best:>13,.0f} terms/s")
    print(f"              ≈ {digits / best:>13,.0f} digits/s")
    print("  (lower time = faster core; throughput is for cross-machine compare)")
    print("=" * 64)


if __name__ == "__main__":
    main()
