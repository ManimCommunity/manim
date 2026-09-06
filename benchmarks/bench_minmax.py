"""Benchmark: np.min+np.max vs numpy_minmax for arrays used in Manim."""

import json
import platform
import time
from pathlib import Path

import numpy as np

print("Python:", platform.python_version())
print("NumPy:", np.__version__)
print("Platform:", platform.platform())
print("Processor:", platform.processor())


try:
    import numpy_minmax

    HAS_MINMAX = True
    print("numpy-minmax:", numpy_minmax.__version__)
except ImportError:
    HAS_MINMAX = False
    print("numpy-minmax not installed. Run: uv add numpy-minmax")


REPS = 500
RESULTS = {}


def bench(label, fn, *args):
    """Run a benchmark with warmup and record the average execution time."""
    # Warmup
    for _ in range(10):
        fn(*args)

    t0 = time.perf_counter()

    for _ in range(REPS):
        fn(*args)

    elapsed = (time.perf_counter() - t0) / REPS * 1e6
    RESULTS[label] = round(elapsed, 3)

    print(f"  {label:55s}: {elapsed:8.3f} us")


def np_minmax_1d(arr):
    return np.min(arr), np.max(arr)


def np_minmax_2d(arr):
    return np.min(arr, axis=0), np.max(arr, axis=0)


def mm_minmax_1d(arr):
    return numpy_minmax.minmax(arr)


rng = np.random.default_rng(100)


# -- 1D float64 ---------------------------------------------------------------
# This represents the precision-preserving production path:
# numpy_minmax.minmax(values)
print("\n[1] 1D float64 (size=10_000) - mobject.py values array")

arr_1d_f64 = rng.random(10_000).astype(np.float64)

bench(
    "np.min + np.max [float64]",
    np_minmax_1d,
    arr_1d_f64,
)

if HAS_MINMAX:
    bench(
        "numpy_minmax.minmax [float64]",
        mm_minmax_1d,
        arr_1d_f64,
    )


# -- 1D float32 ---------------------------------------------------------------
# numpy-minmax is specifically optimized for float32 arrays.
print("\n[2] 1D float32 (size=10_000) - optimized dtype comparison")

arr_1d_f32 = rng.random(10_000).astype(np.float32)

bench(
    "np.min + np.max [float32]",
    np_minmax_1d,
    arr_1d_f32,
)

if HAS_MINMAX:
    bench(
        "numpy_minmax.minmax [float32]",
        mm_minmax_1d,
        arr_1d_f32,
    )


# -- 2D float32 ---------------------------------------------------------------
# Reference benchmark for point/polygon arrays using axis=0.
print("\n[3] 2D float32 (5000x3) - points/polygon arrays (axis=0)")

arr_2d_f32 = rng.random((5_000, 3)).astype(np.float32)

bench(
    "np.min(axis=0) + np.max(axis=0) [float32]",
    np_minmax_2d,
    arr_2d_f32,
)

# numpy_minmax does not provide the same axis=0 operation for multidimensional
# arrays, so it is intentionally not benchmarked here.


# -- 2D float64 ---------------------------------------------------------------
print("\n[4] 2D float64 (5000x3) - points/polygon arrays (axis=0)")

arr_2d_f64 = rng.random((5_000, 3)).astype(np.float64)

bench(
    "np.min(axis=0) + np.max(axis=0) [float64]",
    np_minmax_2d,
    arr_2d_f64,
)


# -- Save results --------------------------------------------------------------
out = Path(__file__).parent / "bench_minmax_results.json"

with open(out, "w") as f:
    json.dump(
        {
            "reps": REPS,
            "unit": "microseconds",
            "results": RESULTS,
        },
        f,
        indent=2,
    )

print(f"\nResults saved -> {out}")
