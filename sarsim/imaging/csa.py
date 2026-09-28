"""Chirp Scaling Algorithm: NumPy/CuPy port of ChirpScalingAlgo::apply_csa
(src/ChirpScalingAlgo.cpp). Matches the C++ focused_image to ~5e-16 relative.

The five phase filters depend only on the imaging parameters, so they are built
once in __init__ (two of the C++ pairs are pre-multiplied exactly as
apply_second_phase_func / apply_third_phase_func do) and reused across scenes.

Filter layout mirrors the C++: every filter is evaluated on the centered
(fftshifted) frequency axis and stored at fft_shift_index(i), i.e. ifftshifted
along each frequency axis -- azimuth (axis 0) for all five, range (axis 1)
additionally for range_comp_filt and second_comp_filt.

FFT normalization matches FFTW's use there: forward unnormalized, inverse / N
(NumPy/CuPy's default "backward" norm).
"""
import numpy as np

from ..params import LIGHT_SPEED_M_S, PI, freq_axis
from .base import ImagingAlgorithm, register


@register("csa")
class ChirpScalingAlgo(ImagingAlgorithm):
    def __init__(self, xp, ax, par):
        super().__init__(xp, ax, par)
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

    def focus(self, echo):
        fft = self.xp.fft
        x = fft.fft(echo, axis=0)          # azimuth FFT
        x *= self.chirp_scaling
        x = fft.fft(x, axis=1)             # range FFT
        x *= self.second_phase
        x = fft.ifft(x, axis=1)            # range IFFT
        x *= self.third_phase
        return fft.ifft(x, axis=0)         # azimuth IFFT
