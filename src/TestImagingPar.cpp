#include "../include/util.h"
#include "../include/util.cpp"
#include "ImagingPar.h"

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
    if (argc == 2 && std::string(argv[1]) == "test_echo_sig")
    {
        save_mat_to_npy("echo_signal_result.npy", imaging_par.gen_point_target_echo_signal(), imaging_par.n_row, imaging_par.n_col);
    }
    return 0;
}