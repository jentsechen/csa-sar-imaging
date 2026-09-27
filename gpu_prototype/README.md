# GPU prototype: `gen_echo_signal` echo-signal generation

Evaluates GPU-accelerating the point-target echo-signal generator
(`ImagingPar::gen_point_target_echo_signal` in `src/ImagingPar.cpp`, called
from `src/apps/gen_echo_signal.cpp`). This is the slowest stage of the
end-to-end union-mask pipeline (`end_to_end_pipeline/gen_echo_signal_union_batch.py`).

## Why this step is a good GPU candidate

For each point target, for each `(azimuth, range)` sample pair, the kernel
computes a slant range and a complex chirp/carrier phase, then accumulates
into an output image. Every `(target, i, j)` triple is independent except for
the final accumulation — a classic embarrassingly-parallel, trig-heavy,
low-memory-bandwidth workload.

## Correctness finding: a pre-existing race condition in `gen_echo_signal`

`ImagingPar::gen_point_target_echo_signal` parallelizes the **outer loop over
point targets** with `OMP_FOR` (`#pragma omp parallel for`), but every thread
writes into the **same shared output array** with an unprotected `+=`. When
two targets' range windows overlap the same output pixel — routine for a
dense scatterer mask — concurrent threads race on that pixel and updates get
lost.

This was confirmed empirically: running the exact same binary on the exact
same input twice produced different output (max abs diff ~470–478 out of a
peak magnitude of ~5.2e4, i.e. ~0.9% relative error), and a single-threaded
(`OMP_NUM_THREADS=1`) run — for which no race is possible — differed from both
multi-threaded runs by the same magnitude.

**Consequence:** don't validate against the default multi-threaded build's
output. Always compare against a `OMP_NUM_THREADS=1` reference run, and be
aware that any of this project's existing echo-signal `.npy` outputs
generated with the default multi-threaded build may be silently wrong.

## Benchmark (measured 2026-09-27, target `P0033_1800_2600_4200_5000`, 2372 point targets, 3200x3200 grid)

| Version | Time | Notes |
|---|---|---|
| C++, single-threaded (`OMP_NUM_THREADS=1`) | 1289.7 s (21m30s) | Correct, race-free — the real baseline |
| C++, default (24-core OpenMP) | ~74 s | **Has the race bug above — result is wrong by ~0.9%** |
| CuPy (this prototype, RTX 5090) | ~6.7 s | Matches the single-threaded reference to ~1e-8 relative (float noise only) |

- Speedup vs. the correct single-threaded baseline: **~192x**
- Speedup vs. the (buggy) default multi-threaded build: **~11x**, while also being correct

## Where the time goes in this prototype (and the obvious next step)

This prototype is a straightforward CuPy port: a Python-level loop over point
targets, each iteration doing several unfused elementwise ops (`sqrt`,
comparisons, `exp`, multiplies, `where`) over the full 3200x3200 grid. Each of
those materializes a full-size intermediate array in GPU memory — across
2372 targets this adds up to several hundred GB–TB of redundant memory
traffic, on top of thousands of individual kernel launches.

A hand-written CUDA kernel where **each thread owns one output pixel** and
loops over the point-target list (kept in constant memory) internally, only
writing the final accumulated value once, would eliminate essentially all of
that redundant memory traffic and the launch overhead. Expected to give a
further significant speedup on top of the ~11x above, though the exact factor
needs to be measured once written. A cheaper way to capture most of that
win without touching the C++/CMake build is a CuPy `RawKernel` (raw CUDA C
compiled via NVRTC from Python) implementing the same thread-per-pixel
design — worth trying before a full native integration into `gen_echo_signal.cpp`.

## Usage

```bash
pip install --user cupy-cuda12x
cd end_to_end_pipeline
python gen_echo_signal_union_batch.py --n 1   # generates the point-target JSON + a C++ reference .npy
python ../gpu_prototype/gen_echo_signal_cupy.py --target <stem>
```

The script validates its GPU output against the C++ `.npy` already on disk
under `union_pipeline/echo_signal/` — regenerate that reference with
`OMP_NUM_THREADS=1` first if you want a guaranteed race-free comparison.
