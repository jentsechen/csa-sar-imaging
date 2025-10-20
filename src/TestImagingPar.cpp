#include "../include/util.h"
#include "../include/util.cpp"
#include "ImagingPar.h"
#include "../cnpy/cnpy.h"

int main()
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
    // echo signal generation
    std::vector<std::vector<std::complex<double>>> single_point_echo_signal = imaging_par.gen_point_target_echo_signal();
    std::cout << "size: (" << single_point_echo_signal.size() << ", " << single_point_echo_signal[0].size() << ")" << std::endl;
    std::cout << single_point_echo_signal[5555][2560] << std::endl;
    cnpy::npz_save("out.npz", "myVar1", &single_point_echo_signal, {1});
    return 0;
}