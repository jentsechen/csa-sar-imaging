// #include "../include/util.h"
// #include "../include/util.cpp"
// #include "ImagingPar.h"
// #include "ChirpScalingAlgo.h"
// #include "../cnpy/cnpy.h"
// #include <chrono>

// void save_mat_to_npy(const std::vector<std::vector<std::complex<double>>> &mat, std::string file_path)
// {
//     std::vector<std::complex<double>> vec = flatten(mat);
//     cnpy::npy_save(file_path, vec.data(), {mat.size(), mat[0].size()}, "w");
// }

// double thresholding(double input)
// {
//     const double threshold = 100;
//     double mag = std::abs(input);
//     if (mag > threshold)
//         return (input > 0) ? (mag - threshold) : (threshold - mag);
//     return 0.0;
// }

// std::complex<double> thresholding(const std::complex<double> &input)
// {
//     return std::complex<double>(thresholding(input.real()), thresholding(input.imag()));
// }

// std::vector<std::complex<double>> thresholding(const std::vector<std::complex<double>> &input)
// {
//     std::vector<std::complex<double>> output(input.size());
//     for (auto i = 0; i < input.size(); i++)
//     {
//         output[i] = thresholding(input[i]);
//     }
//     return output;
// }

// std::vector<std::vector<std::complex<double>>> thresholding(const std::vector<std::vector<std::complex<double>>> &input)
// {
//     std::vector<std::vector<std::complex<double>>> output(input.size());
//     for (auto i = 0; i < input.size(); i++)
//     {
//         output[i] = thresholding(input[i]);
//     }
//     return output;
// }

// int main(int argc, char *argv[])
// {
//     json input_par = load_json("input_par.json");
//     SigPar sig_par(input_par.find("wavelength_m")->get<double>(), input_par.find("pulse_width_sec")->get<double>(),
//                    input_par.find("pulse_rep_freq_hz")->get<double>(), input_par.find("bandwidth_hz")->get<double>(),
//                    input_par.find("sampling_freq_hz")->get<double>());
//     ImagingPar imaging_par(sig_par, input_par.find("closest_slant_range_m")->get<double>());
//     ChirpScalingAlgo chirp_scaling_algo(imaging_par);
//     cnpy::NpyArray echo_sig_npy = cnpy::npy_load("./echo_signal.npy");
//     std::complex<double> *echo_sig_ptr = echo_sig_npy.data<std::complex<double>>();
//     const size_t n_row = chirp_scaling_algo.imaging_par.azimuth_freq_axis_hz.size();
//     const size_t n_col = chirp_scaling_algo.imaging_par.range_time_axis_sec.size();
//     std::vector<std::vector<std::complex<double>>> echo_signal;
//     resize_mat(echo_signal, n_row, n_col);
//     // #pragma omp parallel for
//     for (auto i = 0; i < n_row; i++)
//     {
//         for (auto j = 0; j < n_col; j++)
//         {
//             echo_signal[i][j] = echo_sig_ptr[i * n_col + j];
//         }
//     }
//     std::vector<std::vector<std::complex<double>>> csa_out;
//     resize_mat(csa_out, n_row, n_col);
//     // #pragma omp parallel for
//     for (auto i = 0; i < n_row; i++)
//     {
//         for (auto j = 0; j < n_col; j++)
//         {
//             csa_out[i][j] = std::complex<double>(0.0, 0.0);
//         }
//     }

//     std::cout << "The first iteration" << std::endl;
//     auto start_time = std::chrono::high_resolution_clock::now();
//     std::vector<std::vector<std::complex<double>>> diff_csa_out = chirp_scaling_algo.apply_csa(echo_signal - chirp_scaling_algo.apply_inverse_csa(csa_out));
//     csa_out = csa_out + diff_csa_out;
//     csa_out = thresholding(csa_out);
//     // csa_out = chirp_scaling_algo.apply_csa(echo_signal);
//     auto end_time = std::chrono::high_resolution_clock::now();
//     auto duration = end_time - start_time;
//     auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
//     std::cout << "Execution time: " << milliseconds << " ms" << std::endl;

//     std::cout << "The first save" << std::endl;
//     start_time = std::chrono::high_resolution_clock::now();
//     save_mat_to_npy(csa_out, "./result/csa_out_iter_0.npy");
//     end_time = std::chrono::high_resolution_clock::now();
//     duration = end_time - start_time;
//     milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
//     std::cout << "Execution time: " << milliseconds << " ms" << std::endl;

//     std::cout << "The second iteration" << std::endl;
//     start_time = std::chrono::high_resolution_clock::now();
//     diff_csa_out = chirp_scaling_algo.apply_csa(echo_signal - chirp_scaling_algo.apply_inverse_csa(csa_out));
//     csa_out = csa_out + diff_csa_out;
//     csa_out = thresholding(csa_out);
//     end_time = std::chrono::high_resolution_clock::now();
//     duration = end_time - start_time;
//     milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
//     std::cout << "Execution time: " << milliseconds << " ms" << std::endl;

//     std::cout << "The second save" << std::endl;
//     start_time = std::chrono::high_resolution_clock::now();
//     save_mat_to_npy(csa_out, "./result/csa_out_iter_1.npy");
//     end_time = std::chrono::high_resolution_clock::now();
//     duration = end_time - start_time;
//     milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
//     std::cout << "Execution time: " << milliseconds << " ms" << std::endl;
// }