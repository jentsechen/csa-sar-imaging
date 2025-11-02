#ifndef __chirp_scaling_algo__
#define __chirp_scaling_algo__
#include <complex>
#include <vector>
#include "../include/util.h"
#include "ImagingPar.h"

class ChirpScalingAlgo
{
private:
    void gen_migr_par();
    void modify_range_fm_rate_hz_s();
    void gen_chirp_scaling();

public:
    ImagingPar imaging_par;
    std::vector<double> migr_par;                    // D(f_\eta, V_r)
    std::vector<double> modified_range_fm_rate_hz_s; // K_m
    std::vector<std::vector<std::complex<double>>> chirp_scaling;
    std::vector<std::vector<std::complex<double>>> range_comp_filt, azimuth_comp_filt;
    std::vector<std::vector<std::complex<double>>> second_comp_filt, third_comp_filt;
    ChirpScalingAlgo(const ImagingPar &imaging_par);
};

#endif