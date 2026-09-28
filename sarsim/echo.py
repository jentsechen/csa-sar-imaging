"""Point-target echo generation: hand-written CUDA kernel (CuPy RawKernel,
compiled via NVRTC) for ImagingPar::gen_point_target_echo_signal
(src/ImagingPar.cpp), for azi_win_en=False, noise_en=False,
coherent_scatter_en=False.

One thread per output pixel, accumulating in registers and writing each pixel
exactly once:

  * Targets are sorted by azimuth offset (build_point_target_list emits them
    that way), so the targets whose azimuth window covers row i form a
    contiguous slice [row_lo[i], row_hi[i]) -- computed on the host.
  * Each block covers one azimuth row x blockDim range columns. It walks that
    slice in tiles: each thread computes the slant range / round-trip time /
    carrier*scatter for one target (once per block, not once per pixel), and
    targets whose range window misses the block's columns are dropped via an
    order-preserving (deterministic) block compaction into shared memory.
  * Every thread then only evaluates the chirp for the surviving targets.

Phases use sincospi() (argument pre-divided by pi) -- exact reduction, so the
large carrier phase (~1.8e8 rad) is not hurt by rounding pi*4*R/lambda.
Matches the C++ reference to ~5e-9 relative (see sarsim/validate.py).
"""
import time

import numpy as np

from .params import LIGHT_SPEED_M_S

BLOCK = 256  # threads per block == targets per tile

_KERNEL_SRC = r"""
#define BLOCK %(BLOCK)d
#define N_WARPS (BLOCK / 32)

extern "C" __global__ void __launch_bounds__(BLOCK)
echo_kernel(const double* __restrict__ range_time,
            const double* __restrict__ azimuth_time,
            const double* __restrict__ az_off,
            const double* __restrict__ rg_off,
            const double* __restrict__ coef,
            const int* __restrict__ row_lo,
            const int* __restrict__ row_hi,
            const int n_col, const int row0,
            const double closest_ground_range_m, const double height_m,
            const double sensor_speed_m_s, const double pulse_width_sec,
            const double range_fm_rate_hz_s, const double wavelength_m,
            const double synthetic_aperture_time_sec, const double light_speed_m_s,
            double2* __restrict__ out)
{
    __shared__ double s_rtt[BLOCK];
    __shared__ double s_cr[BLOCK];
    __shared__ double s_ci[BLOCK];
    __shared__ int s_warp_off[N_WARPS + 1];

    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int i = row0 + blockIdx.y;
    const int col0 = blockIdx.x * BLOCK;
    const int j = col0 + tid;
    const int col_last = min(col0 + BLOCK, n_col) - 1;

    const double half_pw = pulse_width_sec / 2.0;
    const double half_sat = synthetic_aperture_time_sec / 2.0;
    const double tau = (j < n_col) ? range_time[j] : 0.0;
    // Conservative block-level range bounds (exact test is repeated per pixel).
    const double slack = 1e-9;
    const double rtt_min = range_time[col0] - half_pw - slack;
    const double rtt_max = range_time[col_last] + half_pw + slack;
    const double t_i = azimuth_time[i];
    const double h2 = height_m * height_m;

    double acc_re = 0.0, acc_im = 0.0;
    const int lo = row_lo[i], hi = row_hi[i];

    for (int base = lo; base < hi; base += BLOCK) {
        const int k = base + tid;
        bool keep = false;
        double rtt = 0.0, cr = 0.0, ci = 0.0;
        if (k < hi) {
            const double a = az_off[k];
            const double rel_az = t_i + a;  // apply_azimuth_window
            if (rel_az < half_sat && rel_az > -half_sat) {
                // calc_slant_range_m, same operation order as the C++.
                const double g = closest_ground_range_m + rg_off[k];
                const double gr = sqrt(h2 + g * g);
                const double v = sensor_speed_m_s * (t_i + a);
                const double slant = sqrt(gr * gr + v * v);
                rtt = 2.0 * slant / light_speed_m_s;
                if (rtt > rtt_min && rtt < rtt_max) {
                    keep = true;
                    double s, c;
                    sincospi(-4.0 * slant / wavelength_m, &s, &c);  // carrier
                    const double sc = coef[k];
                    cr = c * sc;
                    ci = s * sc;
                }
            }
        }
        // Order-preserving block compaction of the kept targets.
        const unsigned ballot = __ballot_sync(0xffffffffu, keep);
        __syncthreads();  // previous tile's shared data fully consumed
        if (lane == 0) s_warp_off[warp + 1] = __popc(ballot);
        __syncthreads();
        if (tid == 0) {
            s_warp_off[0] = 0;
            for (int w = 0; w < N_WARPS; w++) s_warp_off[w + 1] += s_warp_off[w];
        }
        __syncthreads();
        if (keep) {
            const int pos = s_warp_off[warp] + __popc(ballot & ((1u << lane) - 1u));
            s_rtt[pos] = rtt;
            s_cr[pos] = cr;
            s_ci[pos] = ci;
        }
        __syncthreads();
        const int n_kept = s_warp_off[N_WARPS];
        for (int m = 0; m < n_kept; m++) {
            const double rel = tau - s_rtt[m];
            if (rel < half_pw && rel > -half_pw) {  // apply_range_window
                double s, c;
                sincospi(range_fm_rate_hz_s * rel * rel, &s, &c);  // chirp
                const double cr_m = s_cr[m], ci_m = s_ci[m];
                acc_re += c * cr_m - s * ci_m;
                acc_im += c * ci_m + s * cr_m;
            }
        }
    }
    if (j < n_col) out[(size_t)i * n_col + j] = make_double2(acc_re, acc_im);
}
""" % {"BLOCK": BLOCK}

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        import cupy as cp
        _kernel = cp.RawKernel(_KERNEL_SRC, "echo_kernel", options=("-std=c++11",))
    return _kernel


def gen_echo_signal(ax, azimuth_offset_sec, range_offset_m, scatter_coef,
                    deadline=None, rows_per_launch=256):
    """(n_row, n_col) complex128 echo, computed and kept on the GPU.

    Launches the kernel in chunks of `rows_per_launch` azimuth rows so that
    `deadline` (a time.perf_counter() timestamp) can be checked between chunks.
    Returns (output cupy complex128 array, timed_out: bool)."""
    import cupy as cp

    n_row, n_col = ax["n_row"], ax["n_col"]
    az = np.asarray(azimuth_offset_sec, dtype=np.float64)
    rg = np.asarray(range_offset_m, dtype=np.float64)
    sc = np.asarray(scatter_coef, dtype=np.float64)
    order = np.argsort(az, kind="stable")  # identity for build_point_target_list output
    az, rg, sc = az[order], rg[order], sc[order]

    # Per-row contiguous target slice, widened slightly; the kernel repeats the
    # exact apply_azimuth_window test so the widening never changes the result.
    t = np.asarray(ax["azimuth_time_axis_sec"], dtype=np.float64)
    half_sat = ax["synthetic_aperture_time_sec"] / 2.0
    eps = 1e-9
    row_lo = np.searchsorted(az, -half_sat - t - eps, side="left").astype(np.int32)
    row_hi = np.searchsorted(az, half_sat - t + eps, side="right").astype(np.int32)

    d = lambda x: cp.asarray(np.ascontiguousarray(x))
    range_time_d = d(np.asarray(ax["range_time_axis_sec"], dtype=np.float64))
    azimuth_time_d = d(t)
    az_d, rg_d, sc_d = d(az), d(rg), d(sc)
    lo_d, hi_d = d(row_lo), d(row_hi)
    output = cp.zeros((n_row, n_col), dtype=cp.complex128)

    kernel = _get_kernel()
    grid_x = (n_col + BLOCK - 1) // BLOCK
    f64 = np.float64
    for row0 in range(0, n_row, rows_per_launch):
        if deadline is not None:
            cp.cuda.Stream.null.synchronize()
            if time.perf_counter() > deadline:
                return output, True
        n_rows = min(rows_per_launch, n_row - row0)
        kernel((grid_x, n_rows), (BLOCK,), (
            range_time_d, azimuth_time_d, az_d, rg_d, sc_d, lo_d, hi_d,
            np.int32(n_col), np.int32(row0),
            f64(ax["closest_ground_range_m"]), f64(ax["height_m"]),
            f64(ax["sensor_speed_m_s"]), f64(ax["pulse_width_sec"]),
            f64(ax["range_fm_rate_hz_s"]), f64(ax["wavelength_m"]),
            f64(ax["synthetic_aperture_time_sec"]), f64(LIGHT_SPEED_M_S),
            output))
    return output, False
