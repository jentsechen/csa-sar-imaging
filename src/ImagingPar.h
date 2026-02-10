#ifndef __imaging_par__
#define __imaging_par__
#include <iostream>
#include <numeric>
#include <vector>
#include <complex>
#include "../include/util.h"
#include "SigPar.h"
#include "PointTarget.h"
#include "EchoSigGenPar.h"

class ImagingPar
{
private:
    void gen_range_time_axis_sec();
    void gen_azimuth_time_axis_sec();
    double calc_slant_range_m(double azimuth_time_sec, double azimuth_offset_sec, double range_offset_m);
    double calc_round_trip_time_sec(double slant_range_m);
    std::vector<bool> apply_range_window(double round_trip_time_sec);

public:
    SigPar sig_par;
    double closest_slant_range_m;
    double sensor_speed_m_s, azimuth_aperture_len_m;
    double beamwidth_rad, synthetic_aperture_len_m, synthetic_aperture_time_sec;
    size_t n_row, n_col;
    std::vector<double> range_time_axis_sec;   // \tau
    std::vector<double> range_freq_axis_hz;    // f_\tau
    std::vector<double> azimuth_time_axis_sec; // \eta
    std::vector<double> azimuth_freq_axis_hz;  // f_\eta
    EchoSigGenPar echo_sig_gen_par;
    ImagingPar(const SigPar &sig_par,
               const EchoSigGenPar &echo_sig_gen_par,
               double closest_slant_range_m,
               double sensor_speed_m_s = 120,
               double azimuth_aperture_len_m = 1.2);
    std::vector<std::complex<double>> gen_point_target_echo_signal(
        const std::vector<PointTarget> &point_target_list = std::vector<PointTarget>(1, PointTarget()));
};

#endif