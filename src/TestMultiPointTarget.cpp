#include "../include/util.h"
#include "../include/util.cpp"
#include "ImagingPar.h"
#include "ChirpScalingAlgo.h"
#include <random>

double thresholding(double input)
{
    const double threshold = 2;
    double mag = std::abs(input);
    if (mag > threshold)
        return (input > 0) ? (mag - threshold) : (threshold - mag);
    return 0.0;
}

std::complex<double> thresholding(const std::complex<double> &input)
{
    return std::complex<double>(thresholding(input.real()), thresholding(input.imag()));
}

std::vector<std::complex<double>> thresholding(const std::vector<std::complex<double>> &input)
{
    std::vector<std::complex<double>> output(input.size());
    for (auto i = 0; i < input.size(); i++)
    {
        output[i] = thresholding(input[i]);
    }
    return output;
}

std::vector<std::vector<std::complex<double>>> thresholding(const std::vector<std::vector<std::complex<double>>> &input)
{
    std::vector<std::vector<std::complex<double>>> output(input.size());
    for (auto i = 0; i < input.size(); i++)
    {
        output[i] = thresholding(input[i]);
    }
    return output;
}

int main(int argc, char *argv[])
{
    // json input_par = load_json("input_par.json");
    json input_par = load_json("input_par_img.json");
    SigPar sig_par(input_par.find("wavelength_m")->get<double>(), input_par.find("pulse_width_sec")->get<double>(),
                   input_par.find("pulse_rep_freq_hz")->get<double>(), input_par.find("bandwidth_hz")->get<double>(),
                   input_par.find("sampling_freq_hz")->get<double>());
    ImagingPar imaging_par(sig_par, input_par.find("closest_slant_range_m")->get<double>());
    json output_par{{"range_time_axis_sec", imaging_par.range_time_axis_sec},
                    {"range_freq_axis_hz", imaging_par.range_freq_axis_hz},
                    {"azimuth_time_axis_sec", imaging_par.azimuth_time_axis_sec},
                    {"azimuth_freq_axis_hz", imaging_par.azimuth_freq_axis_hz}};
    // save_json("output_par.json", output_par);
    save_json("output_par_img.json", output_par);
    // std::cout << "n_row: " << imaging_par.n_row << std::endl;
    // std::cout << "n_col: " << imaging_par.n_col << std::endl;
    assert(argc == 3);
    if (std::string(argv[1]) == "test_image")
    {
        json point_target_location = load_json("point_target_location.json");
        std::vector<PointTarget> point_target_list;
        std::cout << "n_row: " << imaging_par.n_row << std::endl;
        std::cout << "n_col: " << imaging_par.n_col << std::endl;
        for (auto i = 0; i < imaging_par.n_row; i++)
        {
            for (auto j = 0; j < imaging_par.n_col; j++)
            {
                if (point_target_location[i][j] == 1)
                {
                    auto azimuth_offset_m = [&]()
                    {
                        return (static_cast<double>(i) - imaging_par.n_row / 2.0) / sig_par.pulse_rep_freq_hz;
                    };
                    auto range_offset_m = [&]()
                    {
                        return (static_cast<double>(j) - imaging_par.n_col / 2.0) * sig_par.light_speed_m_s / sig_par.sampling_freq_hz;
                    };
                    point_target_list.emplace_back(PointTarget(azimuth_offset_m(), range_offset_m()));
                }
            }
        }
        std::cout << "number of point target: " << point_target_list.size() << std::endl;
        return 0;
    }
    if (std::string(argv[1]) == "gen")
    {
        if (std::string(argv[2]) == "single_point_target")
        {
            save_mat_to_npy("./echo_signal/single_point_target.npy", imaging_par.gen_point_target_echo_signal(std::vector<PointTarget>({PointTarget(0.0, 0.0)})), imaging_par.n_row, imaging_par.n_col);
        }
        if (std::string(argv[2]) == "multi_point_target")
        {
            // save_mat_to_npy("./echo_signal/multi_point_target.npy", imaging_par.gen_point_target_echo_signal(std::vector<PointTarget>({PointTarget(), PointTarget(0.005, 0.0), PointTarget(0.0, 3.0)})), imaging_par.n_row, imaging_par.n_col);
            json point_target_location = load_json("point_target_location.json");
            std::vector<PointTarget> point_target_list;
            // assert(point_target_location.size() == imaging_par.n_row && point_target_location[0].size() == imaging_par.n_col);
            std::cout << "n_row: " << imaging_par.n_row << std::endl;
            std::cout << "n_col: " << imaging_par.n_col << std::endl;
            for (auto i = 0; i < imaging_par.n_row / 4; i++)
            {
                for (auto j = 0; j < imaging_par.n_col / 4; j++)
                {
                    if (point_target_location[i][j] == 1)
                    {
                        auto azimuth_offset_m = [&]()
                        {
                            return (static_cast<double>(i + imaging_par.n_row * 3 / 8) - imaging_par.n_row / 2.0) / sig_par.pulse_rep_freq_hz;
                        };
                        auto range_offset_m = [&]()
                        {
                            return (static_cast<double>(j + imaging_par.n_col * 3 / 8) - imaging_par.n_col / 2.0) * sig_par.light_speed_m_s / sig_par.sampling_freq_hz;
                        };
                        point_target_list.emplace_back(PointTarget(azimuth_offset_m(), range_offset_m()));
                    }
                }
            }
            std::cout << "number of point target: " << point_target_list.size() << std::endl;
            // save_mat_to_npy("./echo_signal/multi_point_target.npy", imaging_par.gen_point_target_echo_signal(point_target_list), imaging_par.n_row, imaging_par.n_col);
            save_mat_to_npy("./echo_signal/multi_point_target_image.npy", imaging_par.gen_point_target_echo_signal(point_target_list), imaging_par.n_row, imaging_par.n_col);
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
        cnpy::NpyArray arr = cnpy::npy_load(std::string(argv[2]) + ".npy");
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
        save_mat_to_npy(std::string(argv[2]) + "_mag_db.npy", focused_image_mag_db, n_row, n_col);
    }
    if (std::string(argv[1]) == "iter_recov")
    {
        // cnpy::NpyArray echo_sig_npy = cnpy::npy_load("./echo_signal/single_point_target.npy");
        // cnpy::NpyArray echo_sig_npy = cnpy::npy_load("./echo_signal/multi_point_target.npy");
        cnpy::NpyArray echo_sig_npy = cnpy::npy_load("./echo_signal/multi_point_target_image.npy");
        const std::vector<size_t> &shape_vec = echo_sig_npy.shape;
        size_t n_row = shape_vec[0], n_col = shape_vec[1];
        std::complex<double> *echo_signal_ptr = echo_sig_npy.data<std::complex<double>>();
        std::vector<std::complex<double>> echo_signal(echo_signal_ptr, echo_signal_ptr + n_row * n_col);
        std::random_device rd;
        std::mt19937 gen(rd());
        auto gen_down_smp_idx = [&](size_t n_total, size_t n_rand)
        {
            std::uniform_int_distribution<> distrib(0, n_total - 1);
            std::vector<size_t> down_smp_idx(n_rand);
            for (auto i = 0; i < down_smp_idx.size(); i++)
            {
                down_smp_idx[i] = distrib(gen);
                std::cout << down_smp_idx[i] << std::endl;
            }
            return down_smp_idx;
        };
        // std::vector<size_t> row_down_smp_idx = gen_down_smp_idx(11110, 3333);
        std::vector<size_t> row_down_smp_idx = gen_down_smp_idx(555, 167);
        for (size_t row_idx : row_down_smp_idx)
        {
            size_t start_idx = row_idx * n_col;
            for (auto col_idx = 0; col_idx < n_col; col_idx++)
            {
                echo_signal[start_idx + col_idx] = std::complex(0.0);
            }
        }
        // std::vector<size_t> col_down_smp_idx = gen_down_smp_idx(2560, 768);
        std::vector<size_t> col_down_smp_idx = gen_down_smp_idx(555, 167);
        for (size_t col_idx : col_down_smp_idx)
        {
            for (auto row_idx = 0; row_idx < n_row; row_idx++)
            {
                echo_signal[row_idx * n_col + col_idx] = std::complex(0.0);
            }
        }

        ChirpScalingAlgo chirp_scaling_algo(imaging_par);
        std::vector<std::complex<double>> csa_out(n_row * n_col);

        // std::vector<std::complex<double>> diff_csa_out = chirp_scaling_algo.apply_csa(echo_signal - chirp_scaling_algo.apply_inverse_csa(csa_out));
        // csa_out = csa_out + diff_csa_out;
        // csa_out = thresholding(csa_out);
        // // std::vector<std::complex<double>> csa_out = chirp_scaling_algo.apply_csa(echo_signal);
        // // save_mat_to_npy("./iter_result/csa_out_iter_0.npy", csa_out, n_row, n_col);
        // save_mat_to_npy("./iter_result_down_smp/csa_out_iter_0.npy", csa_out, n_row, n_col);

        // diff_csa_out = chirp_scaling_algo.apply_csa(echo_signal - chirp_scaling_algo.apply_inverse_csa(csa_out));
        // csa_out = csa_out + diff_csa_out;
        // csa_out = thresholding(csa_out);
        // // save_mat_to_npy("./iter_result/csa_out_iter_1.npy", csa_out, n_row, n_col);
        // save_mat_to_npy("./iter_result_down_smp/csa_out_iter_1.npy", csa_out, n_row, n_col);

        std::vector<std::complex<double>> diff_csa_out(n_row * n_col);
        for (auto iter_idx = 0; iter_idx < 5; iter_idx++)
        {
            diff_csa_out = chirp_scaling_algo.apply_csa(echo_signal - chirp_scaling_algo.apply_inverse_csa(csa_out));
            csa_out = csa_out + diff_csa_out;
            csa_out = thresholding(csa_out);
            // save_mat_to_npy("./iter_result_multi_point_rng_dpl_anal/csa_out_iter_" + std::to_string(iter_idx) + ".npy", csa_out, n_row, n_col);
            save_mat_to_npy("./iter_result_multi_point_image/csa_out_iter_" + std::to_string(iter_idx) + ".npy", csa_out, n_row, n_col);
        }
    }

    return 0;
}