#include "../include/util.h"
#include "../include/util.cpp"
#include "ImagingPar.h"
#include "ChirpScalingAlgo.h"
#include "../cnpy/cnpy.h"

int main(int argc, char *argv[])
{
    json input_par = load_json("input_par.json");
    SigPar sig_par(input_par.find("wavelength_m")->get<double>(), input_par.find("pulse_width_sec")->get<double>(),
                   input_par.find("pulse_rep_freq_hz")->get<double>(), input_par.find("bandwidth_hz")->get<double>(),
                   input_par.find("sampling_freq_hz")->get<double>());
    ImagingPar imaging_par(sig_par, input_par.find("closest_slant_range_m")->get<double>());
    ChirpScalingAlgo chirp_scaling_algo(imaging_par);
    // cnpy::NpyArray echo_sig_npy = cnpy::npy_load("./echo_signal.npy");
    // std::complex<double> *echo_sig_ptr = echo_sig_npy.data<std::complex<double>>();
    // const size_t n_row = chirp_scaling_algo.imaging_par.azimuth_freq_axis_hz.size();
    // const size_t n_col = chirp_scaling_algo.imaging_par.range_time_axis_sec.size();
    // std::vector<std::vector<std::complex<double>>> echo_signal;
    // resize_mat(echo_signal, n_row, n_col);
    // for (auto i = 0; i < n_row; i++)
    // {
    //     for (auto j = 0; j < n_col; j++)
    //     {
    //         echo_signal[i][j] = echo_sig_ptr[i * n_col + j];
    //     }
    // }
}