import multiprocessing as mp
from heapq import merge
from typing import Iterable, List

_PARALLEL_THRESHOLD = 200_000


def _chunked(arr: list, n: int) -> List[list]:
    """Split arr into n roughly-equal contiguous chunks (no copies beyond slicing)."""
    size, rem = divmod(len(arr), n)
    chunks, start = [], 0
    for i in range(n):
        end = start + size + (1 if i < rem else 0)
        chunks.append(arr[start:end])
        start = end
    return chunks


def parallel_merge_sort(arr: Iterable, *, workers: int | None = None) -> list:
    """
    Stable, parallel merge sort using only the standard library.
    Returns a new sorted list.
    """
    arr = list(arr) if not isinstance(arr, list) else arr
    n = len(arr)

    workers = workers or mp.cpu_count()
    if n < _PARALLEL_THRESHOLD or workers <= 1:
        return sorted(arr)

    chunks = _chunked(arr, workers)

    ctx = mp.get_context(
        "spawn" if mp.get_start_method(allow_none=True) != "fork" else "fork"
    )
    with ctx.Pool(workers) as pool:
        sorted_chunks = pool.map(sorted, chunks)

    return list(merge(*sorted_chunks))


if __name__ == "__main__":
    import random
    import time

    random.seed(0)
    data = [random.random() for _ in range(100_000_000)]

    t0 = time.perf_counter()
    a = sorted(data)
    t1 = time.perf_counter()
    b = parallel_merge_sort(data)
    t2 = time.perf_counter()

    print(f"sorted():           {t1 - t0:.3f}s")
    print(f"parallel_merge_sort: {t2 - t1:.3f}s  ({mp.cpu_count()} workers)")
    assert a == b
