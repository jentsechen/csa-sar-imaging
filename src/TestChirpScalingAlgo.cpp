#include "../include/util.h"
#include "../include/util.cpp"
#include "ImagingPar.h"
#include "ChirpScalingAlgo.h"

void parallel_save_matrices(const ChirpScalingAlgo &algo)
{
    std::thread t1(save_mat_to_npy<std::complex<double>>, "./result/chirp_scaling.npy", std::ref(algo.chirp_scaling), algo.n_row, algo.n_col);
    std::thread t2(save_mat_to_npy<std::complex<double>>, "./result/range_comp_filt.npy", std::ref(algo.range_comp_filt), algo.n_row, algo.n_col);
    std::thread t3(save_mat_to_npy<std::complex<double>>, "./result/second_comp_filt.npy", std::ref(algo.second_comp_filt), algo.n_row, algo.n_col);
    std::thread t4(save_mat_to_npy<std::complex<double>>, "./result/azimuth_comp_filt.npy", std::ref(algo.azimuth_comp_filt), algo.n_row, algo.n_col);
    std::thread t5(save_mat_to_npy<std::complex<double>>, "./result/third_comp_filt.npy", std::ref(algo.third_comp_filt), algo.n_row, algo.n_col);
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();
}

int main(int argc, char *argv[])
{
    json input_par = load_json("../TestImagingPar/input_par.json");
    SigPar sig_par(input_par.find("wavelength_m")->get<double>(), input_par.find("pulse_width_sec")->get<double>(),
                   input_par.find("pulse_rep_freq_hz")->get<double>(), input_par.find("bandwidth_hz")->get<double>(),
                   input_par.find("sampling_freq_hz")->get<double>());
    ImagingPar imaging_par(sig_par, input_par.find("closest_slant_range_m")->get<double>());
    ChirpScalingAlgo chirp_scaling_algo(imaging_par);

    if (std::string(argv[1]) == "par")
    {
        cnpy::npy_save("./result/migr_par.npy", chirp_scaling_algo.migr_par.data(), {chirp_scaling_algo.migr_par.size()}, "w");
        cnpy::npy_save("./result/modified_range_fm_rate_hz_s.npy", chirp_scaling_algo.modified_range_fm_rate_hz_s.data(), {chirp_scaling_algo.modified_range_fm_rate_hz_s.size()}, "w");
        parallel_save_matrices(chirp_scaling_algo);
        // save_mat_to_npy("./result/chirp_scaling.npy", chirp_scaling_algo.chirp_scaling, chirp_scaling_algo.n_row, chirp_scaling_algo.n_col);
        // save_mat_to_npy("./result/range_comp_filt.npy", chirp_scaling_algo.range_comp_filt, chirp_scaling_algo.n_row, chirp_scaling_algo.n_col);
        // save_mat_to_npy("./result/second_comp_filt.npy", chirp_scaling_algo.second_comp_filt, chirp_scaling_algo.n_row, chirp_scaling_algo.n_col);
        // save_mat_to_npy("./result/azimuth_comp_filt.npy", chirp_scaling_algo.azimuth_comp_filt, chirp_scaling_algo.n_row, chirp_scaling_algo.n_col);
        // save_mat_to_npy("./result/third_comp_filt.npy", chirp_scaling_algo.third_comp_filt, chirp_scaling_algo.n_row, chirp_scaling_algo.n_col);
    }
    if (std::string(argv[1]) == "app")
    {
        cnpy::NpyArray echo_sig_npy = cnpy::npy_load("../TestImagingPar/echo_signal_result.npy");
        std::complex<double> *echo_signal_ptr = echo_sig_npy.data<std::complex<double>>();
        std::vector<std::complex<double>> echo_signal(echo_signal_ptr, echo_signal_ptr + chirp_scaling_algo.n_row * chirp_scaling_algo.n_col);

        std::vector<std::complex<double>> azimuth_fft_out = chirp_scaling_algo.apply_azimuth_fft(echo_signal);
        std::vector<std::complex<double>> chirp_scaling_out = chirp_scaling_algo.apply_chirp_scaling(azimuth_fft_out);
        std::vector<std::complex<double>> range_fft_out = chirp_scaling_algo.apply_range_fft(chirp_scaling_out);
        std::vector<std::complex<double>> second_phase_func_out_out = chirp_scaling_algo.apply_second_phase_func(range_fft_out);
        std::vector<std::complex<double>> range_ifft_out = chirp_scaling_algo.apply_range_fft(second_phase_func_out_out, true);
        std::vector<std::complex<double>> third_phase_func_out = chirp_scaling_algo.apply_third_phase_func(range_ifft_out);
        std::vector<std::complex<double>> csa_out = chirp_scaling_algo.apply_azimuth_fft(third_phase_func_out, true);

        save_mat_to_npy("./result/azimuth_fft_out.npy", azimuth_fft_out, chirp_scaling_algo.n_row, chirp_scaling_algo.n_col);
        save_mat_to_npy("./result/chirp_scaling_out.npy", chirp_scaling_out, chirp_scaling_algo.n_row, chirp_scaling_algo.n_col);
        save_mat_to_npy("./result/range_fft_out.npy", range_fft_out, chirp_scaling_algo.n_row, chirp_scaling_algo.n_col);
        save_mat_to_npy("./result/second_phase_func_out.npy", second_phase_func_out_out, chirp_scaling_algo.n_row, chirp_scaling_algo.n_col);
        save_mat_to_npy("./result/range_ifft_out.npy", range_ifft_out, chirp_scaling_algo.n_row, chirp_scaling_algo.n_col);
        save_mat_to_npy("./result/third_phase_func_out.npy", third_phase_func_out, chirp_scaling_algo.n_row, chirp_scaling_algo.n_col);
        save_mat_to_npy("./result/csa_out.npy", csa_out, chirp_scaling_algo.n_row, chirp_scaling_algo.n_col);
    }
    if (std::string(argv[1]) == "inverse_csa")
    {
        cnpy::NpyArray echo_sig_npy = cnpy::npy_load("../TestImagingPar/echo_signal_result.npy");
        std::complex<double> *echo_signal_ptr = echo_sig_npy.data<std::complex<double>>();
        std::vector<std::complex<double>> echo_signal(echo_signal_ptr, echo_signal_ptr + chirp_scaling_algo.n_row * chirp_scaling_algo.n_col);
        std::vector<std::complex<double>> csa_out = chirp_scaling_algo.apply_csa(echo_signal);
        std::vector<std::complex<double>> inverse_csa_out = chirp_scaling_algo.apply_inverse_csa(csa_out);
        save_mat_to_npy("./result/csa_out.npy", csa_out, chirp_scaling_algo.n_row, chirp_scaling_algo.n_col);
        save_mat_to_npy("./result/inverse_csa_out.npy", inverse_csa_out, chirp_scaling_algo.n_row, chirp_scaling_algo.n_col);
    }

    std::cout << "DONE" << std::endl;
    return 0;
}