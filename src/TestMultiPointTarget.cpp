#include "../include/util.h"
#include "../include/util.cpp"
#include "ImagingPar.h"
#include "ChirpScalingAlgo.h"

int main(int argc, char *argv[])
{
    json input_par = load_json("input_par.json");
    SigPar sig_par(input_par.find("wavelength_m")->get<double>(), input_par.find("pulse_width_sec")->get<double>(),
                   input_par.find("pulse_rep_freq_hz")->get<double>(), input_par.find("bandwidth_hz")->get<double>(),
                   input_par.find("sampling_freq_hz")->get<double>());
    ImagingPar imaging_par(sig_par, input_par.find("closest_slant_range_m")->get<double>());
    json output_par{{"range_time_axis_sec", imaging_par.range_time_axis_sec},
                    {"range_freq_axis_hz", imaging_par.range_freq_axis_hz},
                    {"azimuth_time_axis_sec", imaging_par.azimuth_time_axis_sec},
                    {"azimuth_freq_axis_hz", imaging_par.azimuth_freq_axis_hz}};
    save_json("output_par.json", output_par);
    assert(argc == 3);
    if (std::string(argv[1]) == "gen")
    {
        if (std::string(argv[2]) == "single_point_target")
        {
            save_mat_to_npy("./echo_signal/single_point_target.npy", imaging_par.gen_point_target_echo_signal(), imaging_par.n_row, imaging_par.n_col);
        }
        if (std::string(argv[2]) == "multi_point_target")
        {
            save_mat_to_npy("./echo_signal/multi_point_target.npy", imaging_par.gen_point_target_echo_signal(std::vector<PointTarget>({PointTarget(), PointTarget(0.05, 10)})), imaging_par.n_row, imaging_par.n_col);
        }
    }
    if (std::string(argv[1]) == "focus")
    {
        ChirpScalingAlgo chirp_scaling_algo(imaging_par);
        cnpy::NpyArray echo_sig_npy = cnpy::npy_load("./echo_signal/" + std::string(argv[2]) + ".npy");
        std::complex<double> *echo_signal_ptr = echo_sig_npy.data<std::complex<double>>();
        std::vector<std::complex<double>> echo_signal(echo_signal_ptr, echo_signal_ptr + chirp_scaling_algo.n_row * chirp_scaling_algo.n_col);
        std::vector<std::complex<double>> csa_out = chirp_scaling_algo.apply_csa(echo_signal);
        save_mat_to_npy("./focused_image/" + std::string(argv[2]) + ".npy", csa_out, chirp_scaling_algo.n_row, chirp_scaling_algo.n_col);
    }
    if (std::string(argv[1]) == "calc_mag")
    {
        cnpy::NpyArray arr = cnpy::npy_load("./focused_image/" + std::string(argv[2]) + ".npy");
        const std::vector<size_t> &shape_vec = arr.shape;
        assert(shape_vec.size() == 2);
        size_t n_row = shape_vec[0], n_col = shape_vec[1];
        std::complex<double> *ptr = arr.data<std::complex<double>>();
        std::vector<std::complex<double>> focused_image(ptr, ptr + n_row * n_col);
        std::vector<double> focused_image_mag_db(n_row * n_col);
        OMP_FOR
        for (auto i = 0; i < n_row; i++)
        {
            for (auto j = 0; j < n_col; j++)
            {
                focused_image_mag_db[i * n_col + j] = std::log10(square(focused_image[i * n_col + j].real()) + square(focused_image[i * n_col + j].imag()));
            }
        }
        save_mat_to_npy("./focused_image/" + std::string(argv[2]) + "_mag_db.npy", focused_image_mag_db, n_row, n_col);
    }
    return 0;
}