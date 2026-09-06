"""Benchmark: np.min+np.max vs numpy_minmax for arrays used in Manim."""

import json
import time
from pathlib import Path

import numpy as np

try:
    import numpy_minmax

    HAS_MINMAX = True
except ImportError:
    HAS_MINMAX = False
    print("numpy_minmax not installed. Run: pip install numpy-minmax")

REPS = 500
RESULTS = {}


def bench(label, fn, *args):
    # Warmup
    for _ in range(10):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(REPS):
        fn(*args)
    elapsed = (time.perf_counter() - t0) / REPS * 1e6  # microseconds
    RESULTS[label] = round(elapsed, 3)
    print(f"  {label:55s}: {elapsed:8.3f} us")


def np_minmax_1d(arr):
    return np.min(arr), np.max(arr)


def np_minmax_2d(arr):
    return np.min(arr, axis=0), np.max(arr, axis=0)


def mm_minmax_1d(arr):
    return numpy_minmax.minmax(arr)


rng = np.random.default_rng(100)
# -- 1D float32 - matches mobject.py use-case ---------------------------------
print("\n[1] 1D float32 (size=10_000) - mobject.py values array")
arr_1d_f32 = rng.random(10_000).astype(np.float32)
bench("np.min + np.max", np_minmax_1d, arr_1d_f32)
if HAS_MINMAX:
    bench("numpy_minmax.minmax", mm_minmax_1d, arr_1d_f32)

# -- 2D float32 - matches points arrays (n x 3) used in labeled.py / polylabel
print("\n[2] 2D float32 (5000x3) - points/polygon arrays (axis=0)")
arr_2d_f32 = rng.random((5_000, 3)).astype(np.float32)
bench("np.min(axis=0) + np.max(axis=0)", np_minmax_2d, arr_2d_f32)
# numpy_minmax falls back to numpy for ndim>=2, so not tested here

# -- 2D float64 - default numpy dtype -----------------------------------------
print("\n[3] 2D float64 (5000x3) - same but float64")
arr_2d_f64 = rng.random((5_000, 3)).astype(np.float64)
bench("np.min(axis=0) + np.max(axis=0)", np_minmax_2d, arr_2d_f64)

# -- Save results -------------------------------------------------------------
out = Path(__file__).parent / "bench_minmax_results.json"
with open(out, "w") as f:
    json.dump({"reps": REPS, "unit": "microseconds", "results": RESULTS}, f, indent=2)

print(f"\nResults saved -> {out}")
