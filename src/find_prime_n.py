import math
from time import perf_counter

import numpy as np
from numba import njit, prange, set_num_threads


@njit(cache=True)
def base_primes_upto(limit):
    if limit < 3:
        return np.empty(0, dtype=np.int64)

    i = 0
    size = (limit - 1) // 2
    is_comp = np.zeros(size, dtype=np.uint8)

    while True:
        p = 2 * i + 3
        if p * p > limit:
            break
        if is_comp[i] == 0:
            j = (p * p - 3) // 2
            while j < size:
                is_comp[j] = 1
                j += p
        i += 1
    cnt = 0

    for k in range(size):
        if is_comp[k] == 0:
            cnt += 1

    out = np.empty(cnt, dtype=np.int64)
    idx = 0

    for k in range(size):
        if is_comp[k] == 0:
            out[idx] = 2 * k + 3
            idx += 1

    return out


@njit(parallel=True, cache=True, boundscheck=False)
def pi_segmented(n, primes, segment_size):
    if n < 2:
        return 0

    total = 1
    n_eff = n + 1
    n_segs = (n_eff - 2 + segment_size - 1) // segment_size

    for k in prange(n_segs):
        lo = 2 + k * segment_size
        hi = lo + segment_size
        if hi > n_eff:
            hi = n_eff
        if (lo & 1) == 1:
            lo += 1
        if (hi & 1) == 1:
            hi -= 1
        if hi <= lo:
            continue
        seg_len = (hi - lo) // 2
        seg = np.ones(seg_len, dtype=np.uint8)
        for kk in range(primes.shape[0]):
            p = primes[kk]
            psq = p * p
            if psq >= hi:
                break
            if psq >= lo:
                start = psq
            else:
                r = lo % p
                start = lo if r == 0 else lo + (p - r)
                if (start & 1) == 0:
                    start += p
            if start >= hi:
                continue
            idx = (start - lo - 1) // 2
            while idx < seg_len:
                seg[idx] = 0
                idx += p
        c = 0

        for i in range(seg_len):
            c += seg[i]
        total += c

    return total


def pi(n, segment_size=1 << 22):
    set_num_threads(18)
    primes = base_primes_upto(int(math.isqrt(n)))
    return pi_segmented(n, primes, segment_size)


def main():
    n = int(input("Enter n: "))
    start = perf_counter()
    print(pi(n))
    end = perf_counter()
    print(f"Time taken: {end - start:.6f} seconds")


if __name__ == "__main__":
    main()
