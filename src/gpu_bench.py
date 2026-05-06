"""GPU benchmark for Apple Silicon (MPS) — push the GPU to its absolute limit.

Measures:
  - Peak FLOPS in fp32, fp16, bf16 via square matmul (compute-bound)
  - Memory bandwidth via elementwise add (memory-bound)
  - Conv2d throughput (a real ML workload, uses MPSGraph)
  - Sustained throughput over a fixed window (thermal load)
  - Maximum allocatable single tensor on unified memory

Run:  uv run python mps_benchmark.py
"""

import subprocess
from time import perf_counter

import torch

# ---------- system info ----------


def get_system_info() -> None:
    print("=" * 72)
    print("System")
    print("=" * 72)
    try:
        sp = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"], text=True
        )
        for line in sp.splitlines():
            line = line.strip()
            if any(
                k in line
                for k in ("Chipset Model", "Total Number of Cores", "Metal Support")
            ):
                print(f"  {line}")
    except Exception as e:
        print(f"  system_profiler failed: {e}")

    try:
        mem = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        print(f"  Unified Memory:        {int(mem) / 1024**3:.1f} GiB")
    except Exception:
        pass

    print(f"  PyTorch:               {torch.__version__}")
    print(f"  MPS available:         {torch.backends.mps.is_available()}")
    print()


# ---------- helpers ----------


def _sync() -> None:
    torch.mps.synchronize()


def _dt(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


# ---------- compute benchmarks ----------


def benchmark_matmul(
    size: int, dtype: torch.dtype, iters: int = 30, warmup: int = 5
) -> float:
    """Square matmul TFLOPS. Tensors live on GPU; only kernel time is measured."""
    a = torch.randn(size, size, dtype=dtype, device="mps")
    b = torch.randn(size, size, dtype=dtype, device="mps")

    for _ in range(warmup):
        c = a @ b
    _sync()

    start = perf_counter()
    for _ in range(iters):
        c = a @ b
    _sync()
    elapsed = perf_counter() - start

    flops = 2 * size**3 * iters
    tflops = flops / elapsed / 1e12
    print(
        f"  matmul {size:>5}x{size:<5} {_dt(dtype):>8}: "
        f"{elapsed / iters * 1000:7.2f} ms/iter  ->  {tflops:7.2f} TFLOPS"
    )

    del a, b, c
    torch.mps.empty_cache()
    return tflops


# ---------- bandwidth benchmarks ----------


def benchmark_bandwidth(size_mb: int, iters: int = 50) -> float:
    """Memory bandwidth via elementwise add (read a + read b -> write c)."""
    n = (size_mb * 1024 * 1024) // 4  # fp32 element count
    a = torch.randn(n, dtype=torch.float32, device="mps")
    b = torch.randn(n, dtype=torch.float32, device="mps")

    for _ in range(5):
        c = a + b
    _sync()

    start = perf_counter()
    for _ in range(iters):
        c = a + b
    _sync()
    elapsed = perf_counter() - start

    bytes_moved = 3 * n * 4 * iters  # read a, read b, write c
    gbs = bytes_moved / elapsed / 1e9
    print(
        f"  add  {size_mb:>4} MB fp32 x {iters}:  "
        f"{elapsed / iters * 1000:6.2f} ms/iter  ->  {gbs:7.1f} GB/s"
    )

    del a, b, c
    torch.mps.empty_cache()
    return gbs


# ---------- real workload ----------


def benchmark_conv2d(
    batch=64, channels=128, hw=128, kernel=3, iters=30, dtype=torch.float16
) -> None:
    x = torch.randn(batch, channels, hw, hw, dtype=dtype, device="mps")
    w = torch.randn(channels, channels, kernel, kernel, dtype=dtype, device="mps")
    pad = kernel // 2

    for _ in range(5):
        y = torch.nn.functional.conv2d(x, w, padding=pad)
    _sync()

    start = perf_counter()
    for _ in range(iters):
        y = torch.nn.functional.conv2d(x, w, padding=pad)
    _sync()
    elapsed = perf_counter() - start

    # 2 * B * Cout * Hout * Wout * Cin * K * K
    flops = 2 * batch * channels * hw * hw * channels * kernel * kernel * iters
    tflops = flops / elapsed / 1e12
    print(
        f"  conv2d B={batch:<3} C={channels:<3} {hw}x{hw} k={kernel} {_dt(dtype):>8}: "
        f"{elapsed / iters * 1000:6.2f} ms/iter  ->  {tflops:7.2f} TFLOPS"
    )

    del x, w, y
    torch.mps.empty_cache()


# ---------- sustained / thermal ----------


def sustained_load(
    duration_s: float = 30.0, size: int = 8192, dtype: torch.dtype = torch.float16
) -> None:
    """Run matmuls non-stop, sampling TFLOPS each second to expose thermal throttling."""
    a = torch.randn(size, size, dtype=dtype, device="mps")
    b = torch.randn(size, size, dtype=dtype, device="mps")

    for _ in range(3):
        c = a @ b
    _sync()

    print(f"  {size}x{size} {_dt(dtype)} matmul for {duration_s}s")
    print(f"  {'sec':>4} {'TFLOPS':>9}")

    flops_per_iter = 2 * size**3
    t_end = perf_counter() + duration_s
    window_start = perf_counter()
    window_iters = 0
    sample = 0
    peak = 0.0
    samples = []

    while perf_counter() < t_end:
        c = a @ b
        window_iters += 1
        if perf_counter() - window_start >= 1.0:
            _sync()
            elapsed = perf_counter() - window_start
            tflops = flops_per_iter * window_iters / elapsed / 1e12
            sample += 1
            peak = max(peak, tflops)
            samples.append(tflops)
            print(f"  {sample:>4d} {tflops:>9.2f}")
            window_start = perf_counter()
            window_iters = 0

    _sync()
    if samples:
        avg = sum(samples) / len(samples)
        print(
            f"  peak: {peak:.2f} TFLOPS  |  avg: {avg:.2f} TFLOPS  "
            f"|  throttle: {(1 - samples[-1] / peak) * 100:5.1f}% drop from peak"
        )

    del a, b, c
    torch.mps.empty_cache()


# ---------- memory stress ----------


def memory_stress_multi(target_gb_list=(16, 32, 64, 96, 110)):
    """Allocate many ~3 GB fp32 tensors to reach target totals."""
    print("  Allocating many ~3 GB tensors to stress unified memory:")
    for target in target_gb_list:
        tensors = []
        per_chunk_gb = 3
        n_chunks = target // per_chunk_gb
        try:
            for i in range(n_chunks):
                # 3 GB each = 805,306,368 fp32 elements, well under INT_MAX
                t = torch.empty(
                    per_chunk_gb * 1024**3 // 4, dtype=torch.float32, device="mps"
                )
                t.fill_(float(i))
                tensors.append(t)
            torch.mps.synchronize()
            alloc = torch.mps.current_allocated_memory() / 1024**3
            print(
                f"    OK    target {target:>3} GB  ({n_chunks} tensors, "
                f"current_allocated: {alloc:.2f} GB)"
            )
        except (RuntimeError, MemoryError) as e:
            print(
                f"    FAIL  target {target:>3} GB  ({type(e).__name__}: {str(e)[:80]})"
            )
            break
        finally:
            del tensors
            torch.mps.empty_cache()


# ---------- main ----------


def main() -> None:
    if not torch.backends.mps.is_available():
        print("MPS unavailable. Exiting.")
        return

    get_system_info()

    print("=" * 72)
    print("Peak compute  (matmul)")
    print("=" * 72)
    for size in (4096, 8192):
        for dt in (torch.float32, torch.float16, torch.bfloat16):
            benchmark_matmul(size, dt, iters=30)
    print()

    print("=" * 72)
    print("Memory bandwidth  (elementwise add)")
    print("=" * 72)
    for sz in (256, 1024, 4096):
        benchmark_bandwidth(sz, iters=50)
    print()

    print("=" * 72)
    print("Conv2d  (real CNN workload)")
    print("=" * 72)
    benchmark_conv2d(batch=64, channels=128, hw=128, dtype=torch.float16)
    benchmark_conv2d(batch=32, channels=256, hw=64, dtype=torch.float16)
    benchmark_conv2d(batch=16, channels=512, hw=32, dtype=torch.float16)
    print()

    print("=" * 72)
    print("Sustained thermal load")
    print("=" * 72)
    sustained_load(duration_s=30.0, size=8192, dtype=torch.float16)
    print()

    print("=" * 72)
    print("Unified memory stress")
    print("=" * 72)
    memory_stress_multi(target_gb_list=(16, 32, 64, 96, 110))


if __name__ == "__main__":
    main()
