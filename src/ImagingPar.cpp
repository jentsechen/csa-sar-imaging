#include "ImagingPar.h"

ImagingPar::ImagingPar(const SigPar &sig_par,
                       const EchoSigGenPar &echo_sig_gen_par,
                       double closest_slant_range_m,
                       double sensor_speed_m_s,
                       double azimuth_aperture_len_m)
    : sig_par(sig_par), closest_slant_range_m(closest_slant_range_m),
      sensor_speed_m_s(sensor_speed_m_s), azimuth_aperture_len_m(azimuth_aperture_len_m),
      beamwidth_rad(sig_par.wavelength_m / azimuth_aperture_len_m),
      synthetic_aperture_len_m(beamwidth_rad * closest_slant_range_m),
      synthetic_aperture_time_sec(synthetic_aperture_len_m / sensor_speed_m_s),
      echo_sig_gen_par(echo_sig_gen_par)
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
    assert(this->azimuth_time_axis_sec.size() == this->azimuth_freq_axis_hz.size());
    assert(this->range_time_axis_sec.size() == this->range_freq_axis_hz.size());
    this->n_row = this->azimuth_time_axis_sec.size();
    this->n_col = this->range_time_axis_sec.size();
}

void ImagingPar::gen_range_time_axis_sec()
{
    int range_time_axis_sec_len = static_cast<int>(floor(this->echo_sig_gen_par.rng_pad_time * sig_par.pulse_width_sec * sig_par.sampling_freq_hz / 2) * 2);
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

double ImagingPar::calc_slant_range_m(double azimuth_time_sec, double azimuth_offset_sec, double range_offset_m)
{
    return sqrt(square(this->closest_slant_range_m + range_offset_m) + square(this->sensor_speed_m_s * (azimuth_time_sec + azimuth_offset_sec)));
}

double ImagingPar::calc_round_trip_time_sec(double slant_range_m)
{
    return 2.0 * slant_range_m / this->sig_par.light_speed_m_s;
}

std::vector<bool> ImagingPar::apply_range_window(double round_trip_time_sec)
{
    std::vector<bool> output(this->range_time_axis_sec.size());
    for (auto i = 0; i < this->range_time_axis_sec.size(); i++)
    {
        double relative_time = this->range_time_axis_sec[i] - round_trip_time_sec;
        output[i] = (relative_time < this->sig_par.pulse_width_sec / 2.0) && (relative_time > -this->sig_par.pulse_width_sec / 2.0);
    }
    return output;
}

std::vector<std::complex<double>> ImagingPar::gen_point_target_echo_signal(const std::vector<PointTarget> &point_target_list)
{
    size_t azi_n_smp = this->azimuth_time_axis_sec.size();
    size_t rng_n_smp = this->range_time_axis_sec.size();
    std::vector<std::complex<double>> point_target_echo_signal(azi_n_smp * rng_n_smp, std::complex<double>(0.0));
    OMP_FOR
    for (auto target_index = 0; target_index < point_target_list.size(); target_index++)
    {
        for (auto i = 0; i < azi_n_smp; i++)
        {
            double slant_range_m = this->calc_slant_range_m(this->azimuth_time_axis_sec[i], point_target_list[target_index].azimuth_offset_sec, point_target_list[target_index].range_offset_m);
            double round_trip_time_sec = this->calc_round_trip_time_sec(slant_range_m);
            std::vector<bool> range_window = this->apply_range_window(round_trip_time_sec);
            double wa = this->echo_sig_gen_par.azi_win_en ? to_the_fourth(sinc(this->sensor_speed_m_s * this->azimuth_time_axis_sec[i] / this->synthetic_aperture_len_m)) : 1.0;
            for (auto j = 0; j < rng_n_smp; j++)
            {
                auto echo_signal_sample = [&]()
                {
                    if (range_window[j])
                    {
                        std::complex<double> chirp_term = std::exp(std::complex<double>(0.0, PI) * this->sig_par.range_fm_rate_hz_s * square(this->range_time_axis_sec[j] - round_trip_time_sec));
                        std::complex<double> carrier_term = std::exp(std::complex<double>(0.0, -PI) * 4.0 * slant_range_m / this->sig_par.wavelength_m);
                        return chirp_term * carrier_term;
                    }
                    return std::complex<double>(0.0, 0.0);
                };
                point_target_echo_signal[i * rng_n_smp + j] += (echo_signal_sample() * wa);
            }
        }
    }
    return point_target_echo_signal;
}
