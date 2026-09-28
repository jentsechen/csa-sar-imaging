#!/usr/bin/env python3
"""NumPy/CuPy port of ChirpScalingAlgo::apply_csa (src/ChirpScalingAlgo.cpp)
plus TestMultiPointTarget's calc_mag (src/TestMultiPointTarget.cpp).

`xp` is either numpy or cupy; FFTs go to cuFFT under CuPy. The five phase
filters depend only on the imaging parameters, so they are built once per
ChirpScalingAlgo instance (two of the C++ pairs are pre-multiplied exactly as
apply_second_phase_func / apply_third_phase_func do) and reused across scenes.

Filter layout mirrors the C++: every filter is evaluated on the centered
(fftshifted) frequency axis and stored at fft_shift_index(i), i.e. ifftshifted
along each frequency axis -- azimuth (axis 0) for all five, range (axis 1)
additionally for range_comp_filt and second_comp_filt.

FFT normalization matches FFTW's use there: forward unnormalized, inverse / N
(NumPy/CuPy's default "backward" norm).

Usage (validates against a C++ focused_image/<stem>.npy and its _mag_db.npy):
    python csa_cupy.py --target P0033_1800_2600_4200_5000
"""
import argparse
import json
import os
import time

import numpy as np

from gen_echo_signal_cupy import PIPE, LIGHT_SPEED_M_S, build_imaging_axes

PI = 3.14159265358979323846


def freq_axis(sampling_freq_hz, n):
    """ImagingPar's gen_freq_axis: (i - n/2) * fs / n, centered."""
    return (np.arange(n) - n // 2) * (sampling_freq_hz / float(n))


class ChirpScalingAlgo:
    def __init__(self, xp, ax, par):
        self.xp = xp
        c = LIGHT_SPEED_M_S
        v = ax["sensor_speed_m_s"]
        wavelength_m = ax["wavelength_m"]
        carrier_freq_hz = c / wavelength_m
        K = ax["range_fm_rate_hz_s"]
        closest_slant_range_m = par["closest_slant_range_m"]
        n_row, n_col = ax["n_row"], ax["n_col"]
        self.n_row, self.n_col = n_row, n_col

        f_az = xp.asarray(freq_axis(ax["pulse_rep_freq_hz"], n_row))[:, None]   # (n_row, 1)
        f_rg = xp.asarray(freq_axis(ax["sampling_freq_hz"], n_col))[None, :]     # (1, n_col)
        tau_np = np.asarray(ax["range_time_axis_sec"])
        tau = xp.asarray(tau_np)[None, :]                                        # (1, n_col)
        r_tau = tau / 2.0 * 3e8                                                  # range per column

        # gen_migr_par
        D = xp.sqrt(1.0 - ((c * f_az) / (2.0 * v * carrier_freq_hz)) ** 2)
        # modify_range_fm_rate_hz_s
        col_term = (c * f_az ** 2) / (2.0 * v ** 2 * carrier_freq_hz ** 3 * D ** 3)
        Km = K / (1.0 - K * col_term * r_tau)                                    # (n_row, n_col)

        def cexp(phase):
            return xp.exp(1j * phase)

        ishift0 = lambda a: xp.fft.ifftshift(a, axes=0)
        ishift01 = lambda a: xp.fft.ifftshift(a, axes=(0, 1))

        # gen_chirp_scaling
        second_order_col_term = -2.0 / (c * D)
        second_order_term = (tau + second_order_col_term * tau_np[n_col // 2] / 2.0 * 3e8) ** 2
        first_order_term = (1.0 / D - 1.0) * Km
        self.chirp_scaling = ishift0(cexp(PI * first_order_term * second_order_term))
        del first_order_term, second_order_term

        # gen_range_comp_filt * gen_second_comp_filt
        range_comp = cexp(PI * D * (f_rg ** 2 / Km[:, n_col // 2:n_col // 2 + 1]))
        second_comp = cexp((4.0 * PI * closest_slant_range_m / c) * (1.0 / D - 1.0) * f_rg)
        self.second_phase = ishift01(range_comp * second_comp)
        del range_comp, second_comp

        # gen_azimuth_comp_filt * gen_third_comp_filt
        azimuth_comp = cexp((4.0 * PI / wavelength_m) * D * r_tau)
        third_comp = cexp((-4.0 * PI / c ** 2) * (1.0 / D * (1.0 / D - 1.0)) * (r_tau * Km))
        self.third_phase = ishift0(azimuth_comp * third_comp)
        del azimuth_comp, third_comp, Km

    def apply_csa(self, echo):
        fft = self.xp.fft
        x = fft.fft(echo, axis=0)          # azimuth FFT
        x *= self.chirp_scaling
        x = fft.fft(x, axis=1)             # range FFT
        x *= self.second_phase
        x = fft.ifft(x, axis=1)            # range IFFT
        x *= self.third_phase
        return fft.ifft(x, axis=0)         # azimuth IFFT


def calc_mag_db(xp, focused):
    """TestMultiPointTarget calc_mag: 10*log10(re^2 + im^2)."""
    return 10.0 * xp.log10(focused.real ** 2 + focused.imag ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="P0033_1800_2600_4200_5000")
    ap.add_argument("--repeat", type=int, default=5, help="timed CSA repetitions")
    args = ap.parse_args()

    import cupy as cp
    with open(f"{PIPE}/input_par.json") as f:
        par = json.load(f)
    ax = build_imaging_axes(par)

    t0 = time.perf_counter()
    csa = ChirpScalingAlgo(cp, ax, par)
    cp.cuda.Stream.null.synchronize()
    print(f"filter setup (once): {time.perf_counter() - t0:.3f} s")

    echo = cp.asarray(np.load(f"{PIPE}/echo_signal/{args.target}.npy"))
    focused = csa.apply_csa(echo)  # warm-up (cuFFT plan creation)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.repeat):
        focused = csa.apply_csa(echo)
        mag_db = calc_mag_db(cp, focused)
    cp.cuda.Stream.null.synchronize()
    print(f"CSA + calc_mag on GPU: {(time.perf_counter() - t0) / args.repeat * 1e3:.1f} ms/scene")

    ref_path = f"{PIPE}/focused_image/{args.target}.npy"
    if os.path.exists(ref_path):
        ref = np.load(ref_path)
        out = cp.asnumpy(focused)
        rel = np.max(np.abs(out - ref)) / np.max(np.abs(ref))
        print(f"focused vs C++: max abs diff / max|ref| = {rel:.3e}")
        ref_db = np.load(f"{PIPE}/focused_image/{args.target}_mag_db.npy")
        mdb = cp.asnumpy(mag_db)
        peak = np.max(ref_db)
        near = ref_db > peak - 60  # ignore deep nulls where dB is ill-conditioned
        print(f"mag_db vs C++: max abs diff within 60 dB of peak = "
              f"{np.max(np.abs(mdb[near] - ref_db[near])):.3e} dB")


if __name__ == "__main__":
    main()
