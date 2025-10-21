#include "../include/util.h"
#include "../include/util.cpp"
#include "ImagingPar.h"
#include "../cnpy/cnpy.h"

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
    json output_par{{"range_time_axis_sec", imaging_par.range_time_axis_sec},
                    {"range_freq_axis_hz", imaging_par.range_freq_axis_hz},
                    {"azimuth_time_axis_sec", imaging_par.azimuth_time_axis_sec},
                    {"azimuth_freq_axis_hz", imaging_par.azimuth_freq_axis_hz}};
    save_json("output_par.json", output_par);
    std::cout << argv[1] << std::endl;
    if (argc == 2 && std::string(argv[1]) == "test_echo_sig")
    {
        std::cout << "run gen echo sig." << std::endl;
        save_point_target_echo_signal(imaging_par.gen_point_target_echo_signal());
    }
    return 0;
}