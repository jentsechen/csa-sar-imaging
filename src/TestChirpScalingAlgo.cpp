#include "../include/util.h"
#include "../include/util.cpp"
#include "ImagingPar.h"
#include "ChirpScalingAlgo.h"
#include "../cnpy/cnpy.h"

void save_mat_to_npy(const std::vector<std::vector<std::complex<double>>> &mat, std::string file_path)
{
    std::vector<std::complex<double>> vec = flatten(mat);
    cnpy::npy_save(file_path, vec.data(), {mat.size(), mat[0].size()}, "w");
}

int main(int argc, char *argv[])
{
    json input_par = load_json("input_par.json");
    SigPar sig_par(input_par.find("wavelength_m")->get<double>(), input_par.find("pulse_width_sec")->get<double>(),
                   input_par.find("pulse_rep_freq_hz")->get<double>(), input_par.find("bandwidth_hz")->get<double>(),
                   input_par.find("sampling_freq_hz")->get<double>());
    ImagingPar imaging_par(sig_par, input_par.find("closest_slant_range_m")->get<double>());
    ChirpScalingAlgo chirp_scaling_algo(imaging_par);

    cnpy::npy_save("./result/migr_par.npy", chirp_scaling_algo.migr_par.data(), {chirp_scaling_algo.migr_par.size()}, "w");
    cnpy::npy_save("./result/modified_range_fm_rate_hz_s.npy", chirp_scaling_algo.modified_range_fm_rate_hz_s.data(), {chirp_scaling_algo.modified_range_fm_rate_hz_s.size()}, "w");

    save_mat_to_npy(chirp_scaling_algo.chirp_scaling, "./result/chirp_scaling.npy");
    save_mat_to_npy(chirp_scaling_algo.range_comp_filt, "./result/range_comp_filt.npy");
    save_mat_to_npy(chirp_scaling_algo.second_comp_filt, "./result/second_comp_filt.npy");
    save_mat_to_npy(chirp_scaling_algo.azimuth_comp_filt, "./result/azimuth_comp_filt.npy");
    save_mat_to_npy(chirp_scaling_algo.third_comp_filt, "./result/third_comp_filt.npy");

    cnpy::NpyArray echo_sig_npy = cnpy::npy_load("./echo_signal.npy");
    std::complex<double> *echo_sig_ptr = echo_sig_npy.data<std::complex<double>>();
    const size_t n_row = chirp_scaling_algo.imaging_par.azimuth_freq_axis_hz.size();
    const size_t n_col = chirp_scaling_algo.imaging_par.range_time_axis_sec.size();
    std::vector<std::vector<std::complex<double>>> echo_signal;
    resize_mat(echo_signal, n_row, n_col);
    for (auto i = 0; i < n_row; i++)
    {
        for (auto j = 0; j < n_col; j++)
        {
            echo_signal[i][j] = echo_sig_ptr[i * n_col + j];
        }
    }

    std::vector<std::vector<std::complex<double>>> azimuth_fft_out = chirp_scaling_algo.apply_azimuth_fft(echo_signal);
    std::vector<std::vector<std::complex<double>>> chirp_scaling_out = chirp_scaling_algo.apply_chirp_scaling(azimuth_fft_out);
    std::vector<std::vector<std::complex<double>>> range_fft_out = chirp_scaling_algo.apply_range_fft(chirp_scaling_out);
    std::vector<std::vector<std::complex<double>>> second_phase_func_out_out = chirp_scaling_algo.apply_second_phase_func(range_fft_out);
    std::vector<std::vector<std::complex<double>>> range_ifft_out = chirp_scaling_algo.apply_range_fft(second_phase_func_out_out, true);
    std::vector<std::vector<std::complex<double>>> third_phase_func_out = chirp_scaling_algo.apply_third_phase_func(range_ifft_out);
    std::vector<std::vector<std::complex<double>>> csa_out = chirp_scaling_algo.apply_azimuth_fft(third_phase_func_out, true);
    
    save_mat_to_npy(azimuth_fft_out, "./result/azimuth_fft_out.npy");
    save_mat_to_npy(chirp_scaling_out, "./result/chirp_scaling_out.npy");
    save_mat_to_npy(range_fft_out, "./result/range_fft_out.npy");
    save_mat_to_npy(second_phase_func_out_out, "./result/second_phase_func_out.npy");
    save_mat_to_npy(range_ifft_out, "./result/range_ifft_out.npy");
    save_mat_to_npy(third_phase_func_out, "./result/third_phase_func_out.npy");
    save_mat_to_npy(csa_out, "./result/csa_out.npy");

    std::vector<std::vector<std::complex<double>>> csa_out = chirp_scaling_algo.apply_csa(echo_signal);
    std::vector<std::vector<std::complex<double>>> inverse_csa_out = chirp_scaling_algo.apply_inverse_csa(csa_out);
    save_mat_to_npy(csa_out, "./result/csa_out.npy");
    save_mat_to_npy(inverse_csa_out, "./result/inverse_csa_out.npy");

    std::cout << "DONE" << std::endl;
    return 0;
}