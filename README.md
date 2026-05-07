# math-apyhai

Example code for math and performance computation in Python (CPU/GPU benchmarks, π convergence, etc.).

## Requirements

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) for dependency management

## Clone & install

```bash
git clone https://github.com/LoveMig6334/math-apyhai.git
cd math-apyhai
uv sync
```

## Run

```bash
uv run python main.py                       # entry point
uv run python bench/cpu_bench.py [N]        # any benchmark
uv run jupyter lab                          # open the notebooks in jupyter/
```

## `bench/`

| File                  | What it does                                                                                                  |
| --------------------- | ------------------------------------------------------------------------------------------------------------- |
| `cpu_bench.py`        | Multi-core CPU benchmark — renders a Mandelbrot set (default `n=16000`) using `multiprocessing.Pool` and reports real/user/sys time and CPU utilization. |
| `singel_pi_bench.py`  | Single-core CPU benchmark — runs the Chudnovsky π series with binary splitting for a 30 s budget and reports terms/digits per second. |
| `gpu_bench.py`        | Apple Silicon GPU (MPS) benchmark — peak matmul TFLOPS (fp32/fp16/bf16), memory bandwidth, conv2d throughput, sustained thermal load, and unified-memory stress. |
| `cpu-benchmark`       | Pre-built native CPU benchmark binary.                                                                        |
