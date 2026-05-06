from contextlib import closing
from itertools import islice
from os import cpu_count, devnull, times
from sys import argv
from time import perf_counter


def pixels(y, n, abs):
    range7 = bytearray(range(7))
    pixel_bits = bytearray(128 >> pos for pos in range(8))
    c1 = 2.0 / float(n)
    c0 = -1.5 + 1j * y * c1 - 1j
    x = 0
    while True:
        pixel = 0
        c = x * c1 + c0
        for pixel_bit in pixel_bits:
            z = c
            for _ in range7:
                for _ in range7:
                    z = z * z + c
                if abs(z) >= 2.0:
                    break
            else:
                pixel += pixel_bit
            c += c1
        yield pixel
        x += 8


def compute_row(p):
    y, n = p

    result = bytearray(islice(pixels(y, n, abs), (n + 7) // 8))
    result[-1] &= 0xFF << (8 - n % 8)
    return y, result


def ordered_rows(rows, n):
    order = [None] * n
    i = 0
    j = n
    while i < len(order):
        if j > 0:
            row = next(rows)
            order[row[0]] = row
            j -= 1

        if order[i]:
            yield order[i]
            order[i] = None
            i += 1


def compute_rows(n, f):
    row_jobs = ((y, n) for y in range(n))

    if cpu_count() < 2:
        yield from map(f, row_jobs)
    else:
        from multiprocessing import Pool

        with Pool() as pool:
            unordered_rows = pool.imap_unordered(f, row_jobs)
            yield from ordered_rows(unordered_rows, n)


def mandelbrot(n):
    with open(devnull, "wb") as out:
        write = out.write
        with closing(compute_rows(n, compute_row)) as rows:
            write("P4\n{0} {0}\n".format(n).encode())
            for row in rows:
                write(row[1])


DEFAULT_N = 16000


if __name__ == "__main__":
    n = int(argv[1]) if len(argv) > 1 else DEFAULT_N

    t0 = perf_counter()
    times0 = times()

    mandelbrot(n)

    elapsed = perf_counter() - t0
    times1 = times()
    user = (times1.user + times1.children_user) - (times0.user + times0.children_user)
    sys_ = (times1.system + times1.children_system) - (
        times0.system + times0.children_system
    )
    cpu_pct = (user + sys_) / elapsed * 100 if elapsed > 0 else 0

    print(f"n     = {n}")
    print(f"real  = {elapsed:7.3f} s")
    print(f"user  = {user:7.3f} s")
    print(f"sys   = {sys_:7.3f} s")
    print(
        f"cpu   = {cpu_pct:6.1f} %   ({cpu_pct / 100:.2f}x cores, of {cpu_count()} available)"
    )
