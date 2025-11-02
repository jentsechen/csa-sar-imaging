#include "../include/util.h"
#include "../include/util.cpp"
#include "ImagingPar.h"
#include "ChirpScalingAlgo.h"
#include "../cnpy/cnpy.h"

// void save() {}

void save_point_target_echo_signal(const std::vector<std::vector<std::complex<double>>> point_target_echo_signal)
{
    std::cout << "size: (" << point_target_echo_signal.size() << ", " << point_target_echo_signal[0].size() << ")" << std::endl;
    size_t rows = point_target_echo_signal.size(), cols = point_target_echo_signal[0].size();
    size_t total_elements = rows * cols;
    std::vector<std::complex<double>> flat_data;
    flat_data.reserve(total_elements);
    for (const auto &row_vec : point_target_echo_signal)
        flat_data.insert(flat_data.end(), row_vec.begin(), row_vec.end());
    cnpy::npy_save("echo_signal_result.npy", flat_data.data(), {rows, cols}, "w");
}

int main(int argc, char *argv[])
{
    json input_par = load_json("input_par.json");
    SigPar sig_par(input_par.find("wavelength_m")->get<double>(), input_par.find("pulse_width_sec")->get<double>(),
                   input_par.find("pulse_rep_freq_hz")->get<double>(), input_par.find("bandwidth_hz")->get<double>(),
                   input_par.find("sampling_freq_hz")->get<double>());
    ImagingPar imaging_par(sig_par, input_par.find("closest_slant_range_m")->get<double>());
    ChirpScalingAlgo chirp_scaling_algo(imaging_par);

    // cnpy::npy_save("./result/migr_par.npy", chirp_scaling_algo.migr_par.data(), {chirp_scaling_algo.migr_par.size()}, "w");
    // cnpy::npy_save("./result/modified_range_fm_rate_hz_s.npy", chirp_scaling_algo.modified_range_fm_rate_hz_s.data(), {chirp_scaling_algo.modified_range_fm_rate_hz_s.size()}, "w");
    
    std::vector<std::complex<double>> chirp_scaling_vec = flatten(chirp_scaling_algo.chirp_scaling);
    cnpy::npy_save("./result/chirp_scaling.npy", chirp_scaling_vec.data(), {chirp_scaling_algo.chirp_scaling.size(), chirp_scaling_algo.chirp_scaling[0].size()}, "w");

    std::cout << "DONE" << std::endl;

    return 0;
}