#!/usr/bin/env python3
"""
Single-core Chudnovsky π benchmark.

Measures how many Chudnovsky-series terms one CPU core can crunch in a
30-second wall-time budget, using binary splitting on top of CPython's
arbitrary-precision integers. The workload is pure integer arithmetic
and is naturally single-threaded (holds the GIL), so this measures one
core regardless of how many you have. On Apple Silicon, macOS will
schedule it on a performance core automatically.

Usage:
    python3 bench_pi.py
"""

from __future__ import annotations

import sys
import time
from math import exp, isqrt, log

sys.setrecursionlimit(10_000)

# Chudnovsky constants
A = 13_591_409
B = 545_140_134
C = 640_320
C3_OVER_24 = C**3 // 24
DIGITS_PER_TERM = 14.1816474627254776555

BUDGET_S = 30.0  # the budget we report results for
SAFETY = 0.75  # verification run aims for 75 % of the budget
CAL_TIME_S = 4.0  # cap the calibration phase at ~this much wall time


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


def run_terms(N: int) -> float:
    """Run the full Chudnovsky pipeline (binary split + final isqrt + divide)
    for N terms. Returns wall-clock seconds; discards the result."""
    digits = int(N * DIGITS_PER_TERM)
    t0 = time.perf_counter()
    _, Q, T = bs(0, N)
    sqrtC = isqrt(10005 * 10 ** (2 * digits))
    _ = (Q * 426880 * sqrtC) // T
    return time.perf_counter() - t0


def fit_power_law(xs: list[int], ys: list[float]) -> tuple[float, float]:
    """Log-log linear regression. Returns (alpha, c) such that y ≈ c · x^alpha."""
    n = len(xs)
    lx = [log(x) for x in xs]
    ly = [log(y) for y in ys]
    mx, my = sum(lx) / n, sum(ly) / n
    num = sum((lx[i] - mx) * (ly[i] - my) for i in range(n))
    den = sum((lx[i] - mx) ** 2 for i in range(n))
    alpha = num / den
    return alpha, exp(my - alpha * mx)


def main() -> None:
    print("=" * 64)
    print("  Single-core Chudnovsky π benchmark")
    print(f"  Python {sys.version.split()[0]}  ·  {sys.platform}")
    print("=" * 64)

    # ── calibration ────────────────────────────────────────────────
    print(f"\nCalibrating (cap ~{CAL_TIME_S:.0f} s cumulative)...")
    print(f"  {'terms':>10}  {'digits':>12}  {'time (s)':>10}")

    cal_N: list[int] = []
    cal_t: list[float] = []
    N, total = 1000, 0.0
    while total < CAL_TIME_S and N < 5_000_000:
        t = run_terms(N)
        cal_N.append(N)
        cal_t.append(t)
        total += t
        digits = int(N * DIGITS_PER_TERM)
        print(f"  {N:>10,}  {digits:>12,}  {t:>10.4f}")
        # Each doubling of N costs ~2.85× as much time. Stop before
        # the next step would blow well past the cap.
        if total + t * 2.85 > CAL_TIME_S * 1.5:
            break
        N *= 2

    if len(cal_N) < 3:
        print("Not enough calibration points; raise CAL_TIME_S.", file=sys.stderr)
        sys.exit(1)

    # Use only the upper half — those points are in the same large-operand
    # Karatsuba regime that the prediction will land in.
    half = len(cal_N) // 2
    alpha, c = fit_power_law(cal_N[half:], cal_t[half:])
    print(f"\n  Upper-half fit: t ≈ {c:.3e} · N^{alpha:.3f}")

    # ── verification run ──────────────────────────────────────────
    target_t = BUDGET_S * SAFETY
    predicted_N = (int((target_t / c) ** (1.0 / alpha)) // 1000) * 1000

    print(f"\nVerification run: N = {predicted_N:,} (aiming for {target_t:.1f} s)")
    t_actual = run_terms(predicted_N)
    print(
        f"  → {t_actual:.2f} s ({t_actual / BUDGET_S:.0%} of {BUDGET_S:.0f} s budget)"
    )

    # ── headline result ───────────────────────────────────────────
    extrap_N = int((BUDGET_S / c) ** (1.0 / alpha))
    extrap_digits = int(extrap_N * DIGITS_PER_TERM)

    print()
    print("=" * 64)
    print(f"  RESULT: this core in {BUDGET_S:.0f} seconds")
    print("=" * 64)
    print(f"  Chudnovsky terms     ≈ {extrap_N:>15,}")
    print(f"  Decimal digits of π  ≈ {extrap_digits:>15,}")
    print(f"  Throughput           ≈ {extrap_N / BUDGET_S:>11,.0f} terms/s")
    print(f"                       ≈ {extrap_digits / BUDGET_S:>11,.0f} digits/s")
    print("=" * 64)


if __name__ == "__main__":
    main()
