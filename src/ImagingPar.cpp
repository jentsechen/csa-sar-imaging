#include "ImagingPar.h"

ImagingPar::ImagingPar(const SigPar &sig_par, double closest_slant_range_m, double sensor_speed_m_s, double azimuth_aperture_len_m)
    : sig_par(sig_par), closest_slant_range_m(closest_slant_range_m),
      sensor_speed_m_s(sensor_speed_m_s), azimuth_aperture_len_m(azimuth_aperture_len_m),
      beamwidth_rad(sig_par.wavelength_m / azimuth_aperture_len_m),
      synthetic_aperture_len_m(beamwidth_rad * closest_slant_range_m),
      synthetic_aperture_time_sec(synthetic_aperture_len_m / sensor_speed_m_s)
{
    gen_range_time_axis_sec();
    gen_azimuth_time_axis_sec();
    auto gen_freq_axis = [&](double sampling_freq_hz, int n_sample)
    {
        std::vector<double> freq_axis(n_sample);
        for (auto i = 0; i < n_sample; i++)
        {
            freq_axis[i] = (i - n_sample / 2) * (sampling_freq_hz / static_cast<double>(n_sample));
        }
        return freq_axis;
    };
    this->range_freq_axis_hz = gen_freq_axis(sig_par.sampling_freq_hz, static_cast<int>(range_time_axis_sec.size()));
    this->azimuth_freq_axis_hz = gen_freq_axis(sig_par.pulse_rep_freq_hz, static_cast<int>(azimuth_time_axis_sec.size()));
}

void ImagingPar::gen_range_time_axis_sec()
{
    int range_time_axis_sec_len = static_cast<int>(floor(8 * sig_par.pulse_width_sec * sig_par.sampling_freq_hz / 2) * 2);
    this->range_time_axis_sec = std::vector<double>(range_time_axis_sec_len);
    for (auto i = 0; i < range_time_axis_sec_len; i++)
    {
        this->range_time_axis_sec[i] = (double)(i - range_time_axis_sec_len / 2) / sig_par.sampling_freq_hz + 2.0 * closest_slant_range_m / sig_par.light_speed_m_s;
    }
}

void ImagingPar::gen_azimuth_time_axis_sec()
{
    int azimuth_time_axis_sec_len = static_cast<int>(floor(synthetic_aperture_time_sec * sig_par.pulse_rep_freq_hz / 2) * 2);
    this->azimuth_time_axis_sec = std::vector<double>(azimuth_time_axis_sec_len);
    for (auto i = 0; i < azimuth_time_axis_sec_len; i++)
    {
        this->azimuth_time_axis_sec[i] = (double)(i - azimuth_time_axis_sec_len / 2) / sig_par.pulse_rep_freq_hz;
    }
}
