#include "../../include/util.h"
#include "../../include/util.cpp"
#include "../ImagingPar.h"

static void print_usage(const char *prog)
{
    std::cerr << "Usage:" << std::endl;
    std::cerr << "  " << prog << " single_point_target <azimuth_offset_m> <range_offset_m>" << std::endl;
    std::cerr << "  " << prog << " <target_name>   (loads ./point_target_location/<target_name>.json)" << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        print_usage(argv[0]);
        return 1;
    }

    json input_par = load_json("input_par.json");
    SigPar sig_par(input_par.find("wavelength_m")->get<double>(), input_par.find("pulse_width_sec")->get<double>(),
                   input_par.find("pulse_rep_freq_hz")->get<double>(), input_par.find("bandwidth_hz")->get<double>(),
                   input_par.find("sampling_freq_hz")->get<double>());
    EchoSigGenPar echo_sig_gen_par(input_par.find("azi_win_en")->get<bool>(),
                                   input_par.find("rng_pad_time")->get<size_t>(),
                                   input_par.find("noise_en")->get<bool>(),
                                   input_par.find("snr_db")->get<double>(),
                                   input_par.find("coherent_scatter_en")->get<bool>());
    std::cout << input_par.find("height_m")->get<double>() << std::endl;
    ImagingPar imaging_par(sig_par, echo_sig_gen_par, input_par.find("closest_slant_range_m")->get<double>(), input_par.find("height_m")->get<double>());
    json output_par{{"range_time_axis_sec", imaging_par.range_time_axis_sec},
                    {"range_freq_axis_hz", imaging_par.range_freq_axis_hz},
                    {"azimuth_time_axis_sec", imaging_par.azimuth_time_axis_sec},
                    {"azimuth_freq_axis_hz", imaging_par.azimuth_freq_axis_hz}};

    const std::string target = argv[1];

    if (target == "single_point_target")
    {
        if (argc != 4)
        {
            print_usage(argv[0]);
            return 1;
        }
        double azimuth_offset_m = std::stod(argv[2]);
        double range_offset_m   = std::stod(argv[3]);
        save_mat_to_npy("./echo_signal/single_point_target.npy",
                        imaging_par.gen_point_target_echo_signal(std::vector<PointTarget>({PointTarget(azimuth_offset_m, range_offset_m)})),
                        imaging_par.n_row, imaging_par.n_col);
    }
    else
    {
        const std::string json_path = "./point_target_location/" + target + ".json";
        json point_target_location = load_json(json_path);
        std::vector<PointTarget> point_target_list;
        std::cout << "n_row: " << imaging_par.n_row << std::endl;
        std::cout << "n_col: " << imaging_par.n_col << std::endl;
        for (auto i = 0; i < imaging_par.n_row / 4; i++)
        {
            for (auto j = 0; j < imaging_par.n_col / 4; j++)
            {
                if (point_target_location[i][j] > 0)
                {
                    double azimuth_offset_m = (static_cast<double>(i + imaging_par.n_row * 3 / 8) - imaging_par.n_row / 2.0) / sig_par.pulse_rep_freq_hz;
                    double range_offset_m   = (static_cast<double>(j + imaging_par.n_col * 3 / 8) - imaging_par.n_col / 2.0) * sig_par.light_speed_m_s / 2.0 / sig_par.sampling_freq_hz;
                    point_target_list.emplace_back(PointTarget(azimuth_offset_m, range_offset_m, point_target_location[i][j]));
                }
            }
        }
        std::cout << "number of point target: " << point_target_list.size() << std::endl;
        save_mat_to_npy("./echo_signal/" + target + ".npy",
                        imaging_par.gen_point_target_echo_signal(point_target_list),
                        imaging_par.n_row, imaging_par.n_col);
    }

    return 0;
}