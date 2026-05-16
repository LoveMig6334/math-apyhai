"""Measure mouse polling rate with tail-latency awareness.

For an 8000 Hz wireless mouse, the *average* polling rate is a poor metric
because it hides occasional dropouts and jitter. This script reports
percentiles, gap counts (intervals far larger than expected), and can save
raw interval data to CSV for offline plotting.

To compare two USB-receiver positions, run twice with different labels:
  python poll.py --duration 10 --expected-hz 8000 --label left  --out left.csv
  python poll.py --duration 10 --expected-hz 8000 --label right --out right.csv

NOTE on macOS: pynput uses Quartz event taps. The OS may coalesce events,
so the *measured* rate is an upper bound from the OS layer, not the HID
layer. For relative comparison between configurations this is still useful.
For absolute HID-level measurement, use hidapi with the dongle's USB VID/PID.

Grant Accessibility permission first:
  System Settings -> Privacy & Security -> Accessibility
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

from pynput import mouse

DEFAULT_DURATION = 8.0
DEFAULT_WARMUP = 1.0  # discard initial events to avoid Quartz startup jitter


def percentile(sorted_data: list[float], p: float) -> float:
    """Linear-interp percentile, p in [0, 100]. Assumes sorted input."""
    if not sorted_data:
        return float("nan")
    if len(sorted_data) == 1:
        return sorted_data[0]
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


def ascii_histogram(values: list[float], bins: int = 20, width: int = 40) -> str:
    if not values:
        return "(no data)"
    lo, hi = min(values), max(values)
    if lo == hi:
        return f"  all values = {lo:.3f}"
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / step), bins - 1)
        counts[idx] += 1
    peak = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        left = lo + i * step
        right = left + step
        bar = "#" * int(c / peak * width)
        lines.append(f"  [{left:8.2f}, {right:8.2f})  {c:6d}  {bar}")
    return "\n".join(lines)


def collect_events(duration: float, label: str | None) -> list[int]:
    """Capture move-event timestamps for `duration` seconds."""
    timestamps: list[int] = []

    def on_move(_x: int, _y: int) -> None:
        timestamps.append(time.perf_counter_ns())

    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Move the mouse continuously for {duration:.1f}s...")
    listener = mouse.Listener(on_move=on_move)
    listener.start()
    time.sleep(duration)
    listener.stop()
    listener.join()
    return timestamps


def analyze(
    timestamps: list[int],
    expected_hz: int | None,
    warmup_s: float,
) -> dict | None:
    if len(timestamps) < 2:
        return None

    # Drop warmup events relative to the first event
    t0 = timestamps[0]
    warmup_ns = int(warmup_s * 1e9)
    timestamps = [t for t in timestamps if t - t0 >= warmup_ns]
    if len(timestamps) < 2:
        return None

    intervals_us = [(b - a) / 1_000 for a, b in zip(timestamps, timestamps[1:])]
    sorted_us = sorted(intervals_us)

    total_s = (timestamps[-1] - timestamps[0]) / 1e9
    avg_rate = (len(timestamps) - 1) / total_s

    if expected_hz is None:
        # Use the median as the "expected" interval
        expected_us = percentile(sorted_us, 50)
    else:
        expected_us = 1_000_000 / expected_hz

    gaps_2x = sum(1 for x in intervals_us if x > 2 * expected_us)
    gaps_5x = sum(1 for x in intervals_us if x > 5 * expected_us)
    gaps_10x = sum(1 for x in intervals_us if x > 10 * expected_us)

    return {
        "n_events": len(timestamps),
        "duration_s": total_s,
        "avg_rate_hz": avg_rate,
        "intervals_us": intervals_us,
        "min_us": sorted_us[0],
        "p1_us": percentile(sorted_us, 1),
        "p50_us": percentile(sorted_us, 50),
        "mean_us": statistics.fmean(intervals_us),
        "p95_us": percentile(sorted_us, 95),
        "p99_us": percentile(sorted_us, 99),
        "p999_us": percentile(sorted_us, 99.9),
        "max_us": sorted_us[-1],
        "stdev_us": statistics.pstdev(intervals_us),
        "expected_us": expected_us,
        "gaps_2x": gaps_2x,
        "gaps_5x": gaps_5x,
        "gaps_10x": gaps_10x,
    }


def print_report(stats: dict | None, label: str | None) -> None:
    if not stats:
        print("Not enough events. Did you move the mouse?")
        print("On macOS, check Accessibility permissions for your terminal.")
        return

    title = f"=== Results{f' [{label}]' if label else ''} ==="
    print()
    print(title)
    print(f"Events (post-warmup) : {stats['n_events']}")
    print(f"Elapsed              : {stats['duration_s']:.3f} s")
    print(f"Average rate         : {stats['avg_rate_hz']:8.1f} Hz")
    print()
    print("Interval percentiles (microseconds):")
    print(f"  min   : {stats['min_us']:8.2f}   ({1e6 / stats['min_us']:7.0f} Hz inst.)")
    print(f"  p1    : {stats['p1_us']:8.2f}")
    print(f"  p50   : {stats['p50_us']:8.2f}   ({1e6 / stats['p50_us']:7.0f} Hz inst.)")
    print(f"  mean  : {stats['mean_us']:8.2f}")
    print(f"  p95   : {stats['p95_us']:8.2f}")
    print(f"  p99   : {stats['p99_us']:8.2f}")
    print(f"  p99.9 : {stats['p999_us']:8.2f}")
    print(f"  max   : {stats['max_us']:8.2f}   ({1e6 / stats['max_us']:7.0f} Hz inst.)")
    print(f"  stdev : {stats['stdev_us']:8.2f}")
    print()
    print(f"Gap counts (expected interval = {stats['expected_us']:.1f} us):")
    print(f"  > 2x expected  : {stats['gaps_2x']}")
    print(f"  > 5x expected  : {stats['gaps_5x']}")
    print(f"  > 10x expected : {stats['gaps_10x']}")
    print()
    print("Histogram of intervals (us):")
    print(ascii_histogram(stats["intervals_us"], bins=20, width=40))


def save_csv(intervals_us: list[float], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["interval_us"])
        for v in intervals_us:
            w.writerow([f"{v:.3f}"])
    print(f"\nRaw intervals saved to: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Measure mouse polling rate with tail-latency stats."
    )
    ap.add_argument(
        "-d",
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help=f"Capture duration in seconds (default {DEFAULT_DURATION})",
    )
    ap.add_argument(
        "-w",
        "--warmup",
        type=float,
        default=DEFAULT_WARMUP,
        help=f"Initial seconds to discard (default {DEFAULT_WARMUP})",
    )
    ap.add_argument(
        "-e",
        "--expected-hz",
        type=int,
        default=None,
        help="Expected polling rate (e.g. 8000). Used to flag gaps. "
        "Inferred from median if omitted.",
    )
    ap.add_argument(
        "-l",
        "--label",
        type=str,
        default=None,
        help="Label for this run (e.g. 'left-port')",
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Write raw intervals to CSV for offline plotting",
    )
    args = ap.parse_args()

    timestamps = collect_events(args.duration, args.label)
    stats = analyze(timestamps, args.expected_hz, args.warmup)
    print_report(stats, args.label)
    if args.out and stats:
        save_csv(stats["intervals_us"], args.out)


if __name__ == "__main__":
    main()
