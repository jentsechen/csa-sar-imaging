# GPU prototype: `gen_echo_signal` echo-signal generation

> **Status:** the work here has graduated into [`sarsim/`](../sarsim/README.md)
> (CUDA echo kernel → `sarsim/echo.py`, CuPy CSA → `sarsim/imaging/csa.py`,
> batch driver → `python -m sarsim.pipeline`). This directory keeps the
> vectorized CuPy reference (`gen_echo_signal_cupy.py`) and the history /
> measurements below.

Evaluates GPU-accelerating the point-target echo-signal generator
(`ImagingPar::gen_point_target_echo_signal` in `src/ImagingPar.cpp`, called
from `src/apps/gen_echo_signal.cpp`). This was the slowest stage of the
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

## Benchmark: vectorized CuPy (measured 2026-09-28, target `P0033_1800_2600_4200_5000`, 2372 point targets, 3200x3200 grid, real PRF=1662.0375 + azi_pad_time=4)

| Version | Time | Notes |
|---|---|---|
| C++, default (24-core OpenMP) | 58.1 s | Correct (race fixed); azimuth window also skips out-of-view work |
| CuPy (`gen_echo_signal_cupy.py`, RTX 5090) | 6.8 s | Matches the C++ reference to ~1.1e-8 relative (float noise only) |

- **Speedup: ~8.5x**, down from the ~11x measured before the azimuth-window
  fix — expected, since the window now makes the C++ side skip a lot of
  previously-wasted work too, not because the GPU got slower (6.7s -> 6.8s,
  essentially unchanged).

### Where the time goes in the vectorized version

A Python-level loop over point targets, each iteration doing several unfused
elementwise ops (`sqrt`, comparisons, `exp`, multiplies, `where`) over the
full 3200x3200 grid. Each materializes a full-size intermediate array in GPU
memory — across 2372 targets that is several hundred GB–TB of redundant memory
traffic, plus thousands of kernel launches — and ~15/16 of the work is spent
on samples outside each target's window that are zero anyway. This motivated
the hand-written kernel below.

## Hand-written CUDA kernel (now `sarsim/echo.py`)

A CuPy `RawKernel` (NVRTC-compiled, no C++/CMake changes). Each thread owns
one output pixel and accumulates in registers, writing once. Two prunings
avoid wasted work:

- Targets are sorted by azimuth offset, so the targets whose azimuth window
  covers row `i` are a contiguous slice `[row_lo[i], row_hi[i])` (host
  `searchsorted`; the kernel repeats the exact window test).
- Each block (one row x 256 columns) walks that slice in tiles: slant range,
  round-trip time and carrier x scatter are computed once per (row, target)
  per block, and targets whose range window misses the block's columns are
  dropped with an order-preserving (deterministic) shared-memory compaction.

Phases use `sincospi` (exact argument reduction for the ~1.8e8 rad carrier).
Matches the C++ reference to ~8e-9 relative (`python -m sarsim.validate`).

| Scene | Targets | CuPy vectorized | CUDA kernel |
|---|---|---|---|
| `P0033_1800_2600_4200_5000` | 2372 | 6.86 s | **0.10 s** (68x) |
| `P0126_4200_5000_5400_6200` | 52855 | >120 s (timed out) | **2.14 s** |
| `P0131_12000_12800_10200_11000` | 120788 | >120 s (timed out) | **4.77 s** |

### Full batch (1593 scenes, measured 2026-09-28)

| Run | Wall time |
|---|---|
| CuPy vectorized (from `union_pipeline/echo_signal_timing_cupy.csv`) | 4.3 h (4 scenes hit the 120 s cap) |
| CUDA kernel, echo only, nothing written | **4.3 min** (221 s GPU, avg 0.14 s/scene) |
| CUDA kernel, writing every `echo_signal/*.npy` | disk-bound: hours |

The last row is not a GPU limit: `/home` is a Seagate ST4000DM004 (SMR hard
drive) that sustains only ~20 MB/s of large writes, and the full echo set is
~261 GB. Hence `sarsim` never writes the echo: it is regenerated on demand and
passed to the imaging algorithm on the GPU. (The per-scene echo / focused
`.npy` directories under `union_pipeline/` were deleted on 2026-09-28, ~670 GB;
two C++ scenes remain in `union_pipeline/reference/` for regression checks.)

## CuPy CSA (now `sarsim/imaging/csa.py`)

NumPy/CuPy port of `ChirpScalingAlgo::apply_csa` + `calc_mag`. The five phase
filters are built once and reused for every scene; FFTs run on cuFFT.

- Matches the C++ `focused_image` to **5-8e-16 relative** (double rounding),
  `_mag_db` to ~3e-12 dB, on the two reference scenes.
- **8-35 ms/scene** on the GPU for CSA + magnitude.
- Full echo → CSA → JPG over all 1593 scenes: **~4.7 min**
  (`python -m sarsim.pipeline --run baseline_csa`).

Against the C++ pipeline's `union_pipeline/csa_jpg/`: most scenes are
bit-identical and the rest differ by at most a few gray levels (the ~1e-8 echo
differences flip an occasional uint8 truncation, and JPEG spreads it to
neighbouring pixels), except **`P0062_3500_4300_1800_2600`**, where the *C++*
output is corrupt: its focused crop was a flat 206.94 dB (+-0.0002 dB), so the
JPG is normalized noise (mean 124 vs ~0.3 for its neighbours). The CuPy CSA on
the same echo gives a normal image. Most likely silent corruption from the
faulty CPU core below; any evaluation using `csa_jpg/` includes this image.

## CPU 1 (core 4) is faulty -- pin jobs away from it

All 1285 segfault/GPF kernel-log records from the last 3 days (as of
2026-09-28) that name a CPU point to **CPU 1 (core 4)**, across both
`TestMultiPointTarget` and Python itself (plus a `[Hardware Error]` record).
This is the cause of the "~15-20% intermittent segfault" in
`csa_to_jpg_union_batch.py`, and it also crashed a long Python batch.
`sarsim` pins itself away from it (`--exclude-cpus`, default `1`); run other
long jobs with:

```bash
taskset -c 0,2-23 python <script>.py ...
```

## Usage (vectorized reference)

```bash
pip install --user cupy-cuda12x
python gpu_prototype/gen_echo_signal_cupy.py --target P0033_1800_2600_4200_5000
```

Validates against the C++ echo in `union_pipeline/reference/echo_signal/`.
