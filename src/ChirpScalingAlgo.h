#ifndef __chirp_scaling_algo__
#define __chirp_scaling_algo__
#include <complex>
#include <vector>
#include "../include/util.h"
#include "ImagingPar.h"
#include <fftw3.h>

class ChirpScalingAlgo
{
private:
    void gen_migr_par();
    void modify_range_fm_rate_hz_s();
    size_t fft_shift_index(size_t index, size_t half_axis_len) { return (index < half_axis_len) ? (index + half_axis_len) : (index - half_axis_len); }
    void gen_chirp_scaling();
    void gen_range_comp_filt();
    void gen_second_comp_filt();
    void gen_azimuth_comp_filt();
    void gen_third_comp_filt();

public:
    ImagingPar imaging_par;
    std::vector<double> migr_par;                    // D(f_\eta, V_r)
    std::vector<double> modified_range_fm_rate_hz_s; // K_m
    std::vector<std::vector<std::complex<double>>> chirp_scaling;
    std::vector<std::vector<std::complex<double>>> range_comp_filt, azimuth_comp_filt;
    std::vector<std::vector<std::complex<double>>> second_comp_filt, third_comp_filt;
    std::vector<std::vector<std::complex<double>>> apply_azimuth_fft(const std::vector<std::vector<std::complex<double>>> &input, bool is_ifft = false);
    std::vector<std::vector<std::complex<double>>> apply_range_fft(const std::vector<std::vector<std::complex<double>>> &input, bool is_ifft = false);
    std::vector<std::vector<std::complex<double>>> apply_chirp_scaling(const std::vector<std::vector<std::complex<double>>> &input, bool is_conj = false);
    std::vector<std::vector<std::complex<double>>> apply_second_phase_func(const std::vector<std::vector<std::complex<double>>> &input, bool is_conj = false);
    std::vector<std::vector<std::complex<double>>> apply_third_phase_func(const std::vector<std::vector<std::complex<double>>> &input, bool is_conj = false);
    std::vector<std::vector<std::complex<double>>> apply_csa(const std::vector<std::vector<std::complex<double>>> &input);
    std::vector<std::vector<std::complex<double>>> apply_inverse_csa(const std::vector<std::vector<std::complex<double>>> &input);
    ChirpScalingAlgo(const ImagingPar &imaging_par);
};

#endif