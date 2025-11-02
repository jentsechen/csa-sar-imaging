#include "ChirpScalingAlgo.h"

ChirpScalingAlgo::ChirpScalingAlgo(const ImagingPar &imaging_par) : imaging_par(imaging_par)
{
    gen_migr_par();
    modify_range_fm_rate_hz_s();
    gen_chirp_scaling();
}

void ChirpScalingAlgo::gen_migr_par()
{
    this->migr_par.resize(this->imaging_par.azimuth_freq_axis_hz.size());
    for (auto i = 0; i < this->imaging_par.azimuth_freq_axis_hz.size(); i++)
    {
        double num = this->imaging_par.sig_par.light_speed_m_s * this->imaging_par.azimuth_freq_axis_hz[i];
        double den = 2.0 * this->imaging_par.sensor_speed_m_s * this->imaging_par.sig_par.carrier_freq_hz;
        this->migr_par[i] = sqrt(1.0 - square(num / den));
    }
}

void ChirpScalingAlgo::modify_range_fm_rate_hz_s()
{
    this->modified_range_fm_rate_hz_s.resize(this->imaging_par.azimuth_freq_axis_hz.size());
    for (auto i = 0; i < this->imaging_par.azimuth_freq_axis_hz.size(); i++)
    {
        double num = this->imaging_par.sig_par.light_speed_m_s * this->imaging_par.closest_slant_range_m * square(this->imaging_par.azimuth_freq_axis_hz[i]);
        double den = 2.0 * square(this->imaging_par.sensor_speed_m_s) * cube(this->imaging_par.sig_par.carrier_freq_hz) * cube(this->migr_par[i]);
        this->modified_range_fm_rate_hz_s[i] = this->imaging_par.sig_par.range_fm_rate_hz_s / (1.0 - this->imaging_par.sig_par.range_fm_rate_hz_s * num / den);
    }
}

void ChirpScalingAlgo::gen_chirp_scaling()
{
    size_t n_row = this->imaging_par.azimuth_freq_axis_hz.size();
    size_t n_col = this->imaging_par.range_time_axis_sec.size();
    size_t half_n_row = n_row / 2;
    resize_mat(this->chirp_scaling, n_row, n_col);
    for (auto i = 0; i < n_row; i++)
    {
        size_t row_w_idx = (i < half_n_row) ? (i + half_n_row) : (i - half_n_row);
        for (auto j = 0; j < n_col; j++)
        {
            double first_order_term = this->modified_range_fm_rate_hz_s[i] * (1.0 / this->migr_par[i] - 1.0);
            double second_order_term = square(this->imaging_par.range_time_axis_sec[j] - 2.0 * this->imaging_par.closest_slant_range_m / (this->imaging_par.sig_par.light_speed_m_s * this->migr_par[i]));
            chirp_scaling[row_w_idx][j] = std::exp(std::complex<double>(0.0, PI) * first_order_term * second_order_term);
        }
    }
}

// first_order_col_term = obj.modified_range_fm_rate_hz_s.*(1 ./ obj.migr_par - 1);
// second_order_row_term = obj.imaging_par.range_time_axis_sec;
// second_order_col_term = 2 * obj.imaging_par.closest_slant_range_m./ ...(obj.imaging_par.sig_par.light_speed_m_s * obj.migr_par);
// col_len = length(obj.imaging_par.azimuth_freq_axis_hz);
// row_len = length(obj.imaging_par.range_time_axis_sec);
// first_order_mat = repmat(first_order_col_term, 1, row_len);
// second_order_mat = (repmat(second_order_row_term, col_len, 1) - ... repmat(second_order_col_term, 1, row_len)).^ 2;
// chirp_scaling = exp(1j * pi * first_order_mat.*second_order_mat);
// chirp_scaling = fftshift(chirp_scaling, 1);
