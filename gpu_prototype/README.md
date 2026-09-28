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

## Correctness findings (both since fixed)

Two real bugs were found and fixed in `src/ImagingPar.cpp` while building this
prototype; this script was updated to match both.

**1. A data race in `gen_point_target_echo_signal`'s OpenMP parallelization.**
It used to parallelize the **outer loop over point targets** with `OMP_FOR`,
but every thread wrote into the **same shared output array** with an
unprotected `+=`. When two targets' range windows overlapped the same output
pixel — routine for a dense scatterer mask — concurrent threads raced on that
pixel and updates got lost (confirmed empirically: running the same binary
twice on the same input gave different output, ~0.9% relative error). Fixed
by parallelizing the inner azimuth-row loop instead (each thread then owns a
disjoint set of output rows). The default multi-threaded build's output is a
valid reference again.

**2. Azimuth had no genuine zero-padding.** Range has `rng_pad_time`, which
extends its axis while keeping the sample rate fixed, and `apply_range_window`
gives each target a hard, bounded nonzero region — real zero-padding. Azimuth
had neither: `n_row` was driven directly by `pulse_rep_freq_hz` with no
separate padding factor, and with `azi_win_en=False` every target contributed
at full amplitude across the *entire* axis. `EchoSigGenPar::azi_pad_time` and
`ImagingPar::apply_azimuth_window()` now mirror range's mechanism, and
`end_to_end_pipeline` switched to the real reference PRF (1662.0375 Hz,
previously inflated 4x to 6648.15 Hz purely to make `n_row` land on 3200).

## Benchmark (measured 2026-09-28, target `P0033_1800_2600_4200_5000`, 2372 point targets, 3200x3200 grid, real PRF=1662.0375 + azi_pad_time=4)

| Version | Time | Notes |
|---|---|---|
| C++, default (24-core OpenMP) | 58.1 s | Correct (race fixed); azimuth window also skips out-of-view work |
| CuPy (this prototype, RTX 5090) | 6.8 s | Matches the C++ reference to ~1.1e-8 relative (float noise only) |

- **Speedup: ~8.5x**, down from the ~11x measured before the azimuth-window
  fix — expected, since the window now makes the C++ side skip a lot of
  previously-wasted work too, not because the GPU got slower (6.7s -> 6.8s,
  essentially unchanged).

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
under `union_pipeline/echo_signal/`.
