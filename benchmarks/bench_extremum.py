"""Benchmark get_extremum_along_dim min/max implementations."""

import time

import numpy as np
import numpy_minmax

REPS = 10_000


def bench(label, fn, values):
    for _ in range(100):
        fn(values)

    start = time.perf_counter()

    for _ in range(REPS):
        fn(values)

    elapsed = (time.perf_counter() - start) / REPS * 1e6
    print(f"{label:35s}: {elapsed:8.3f} us")


def numpy_extremum(values):
    return np.min(values), np.max(values)


def numpy_minmax_extremum(values):
    return numpy_minmax.minmax(values)


rng = np.random.default_rng(100)

for size in [32, 100, 1_000, 10_000]:
    values = rng.random(size).astype(np.float64)

    print(f"\nArray size: {size}")

    bench("np.min + np.max", numpy_extremum, values)
    bench("numpy_minmax.minmax", numpy_minmax_extremum, values)
