#include "ChirpScalingAlgo.h"

ChirpScalingAlgo::ChirpScalingAlgo(const ImagingPar &imaging_par) : imaging_par(imaging_par)
{
    gen_migr_par();
    modify_range_fm_rate_hz_s();
    gen_chirp_scaling();
    gen_range_comp_filt();
    gen_second_comp_filt();
    gen_azimuth_comp_filt();
    gen_third_comp_filt();
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
    const size_t n_row = this->imaging_par.azimuth_freq_axis_hz.size();
    const size_t n_col = this->imaging_par.range_time_axis_sec.size();
    const size_t half_n_row = n_row / 2;
    resize_mat(this->chirp_scaling, n_row, n_col);
    std::complex<double> common_term(0.0, PI);
    for (auto i = 0; i < n_row; i++)
    {
        size_t i_sft = fft_shift_index(i, half_n_row);
        double first_order_col_term = this->modified_range_fm_rate_hz_s[i] * (1.0 / this->migr_par[i] - 1.0);
        double second_order_col_term = -2.0 * this->imaging_par.closest_slant_range_m / (this->imaging_par.sig_par.light_speed_m_s * this->migr_par[i]);
        for (auto j = 0; j < n_col; j++)
        {
            double first_order_term = first_order_col_term;
            double second_order_term = square(this->imaging_par.range_time_axis_sec[j] + second_order_col_term);
            chirp_scaling[i_sft][j] = std::exp(common_term * first_order_term * second_order_term);
        }
    }
}

void ChirpScalingAlgo::gen_range_comp_filt()
{
    const size_t n_row = this->imaging_par.azimuth_freq_axis_hz.size();
    const size_t n_col = this->imaging_par.range_freq_axis_hz.size();
    const size_t half_n_row = n_row / 2, half_n_col = n_col / 2;
    resize_mat(this->range_comp_filt, n_row, n_col);
    std::complex<double> cmmon_term(0.0, PI);
    for (auto i = 0; i < n_row; i++)
    {
        size_t i_sft = fft_shift_index(i, half_n_row);
        double col_term = this->migr_par[i] / this->modified_range_fm_rate_hz_s[i];
        for (auto j = 0; j < n_col; j++)
        {
            size_t j_sft = fft_shift_index(j, half_n_col);
            double row_term = square(this->imaging_par.range_freq_axis_hz[j]);
            this->range_comp_filt[i_sft][j_sft] = std::exp(cmmon_term * col_term * row_term);
        }
    }
}

void ChirpScalingAlgo::gen_second_comp_filt()
{
    const size_t n_row = this->imaging_par.azimuth_freq_axis_hz.size();
    const size_t n_col = this->imaging_par.range_freq_axis_hz.size();
    const size_t half_n_row = n_row / 2, half_n_col = n_col / 2;
    resize_mat(this->second_comp_filt, n_row, n_col);
    std::complex<double> common_term(0.0, 4.0 * PI * this->imaging_par.closest_slant_range_m / this->imaging_par.sig_par.light_speed_m_s);
    for (auto i = 0; i < n_row; i++)
    {
        size_t i_sft = fft_shift_index(i, half_n_row);
        double col_term = 1.0 / this->migr_par[i] - 1.0;
        for (auto j = 0; j < n_col; j++)
        {
            size_t j_sft = fft_shift_index(j, half_n_col);
            double row_term = this->imaging_par.range_freq_axis_hz[j];
            this->second_comp_filt[i_sft][j_sft] = std::exp(common_term * col_term * row_term);
        }
    }
}

void ChirpScalingAlgo::gen_azimuth_comp_filt()
{
    const size_t n_row = this->imaging_par.azimuth_freq_axis_hz.size();
    const size_t n_col = this->imaging_par.range_time_axis_sec.size();
    const size_t half_n_row = n_row / 2;
    resize_mat(this->azimuth_comp_filt, n_row, n_col);
    std::complex<double> common_term(0.0, 4.0 * PI * this->imaging_par.closest_slant_range_m / this->imaging_par.sig_par.wavelength_m);
    for (auto i = 0; i < n_row; i++)
    {
        size_t i_sft = fft_shift_index(i, half_n_row);
        double col_term = this->migr_par[i];
        for (auto j = 0; j < n_col; j++)
        {
            double row_term = 1.0;
            this->azimuth_comp_filt[i_sft][j] = std::exp(common_term * col_term * row_term);
        }
    }
}

void ChirpScalingAlgo::gen_third_comp_filt()
{
    const size_t n_row = this->imaging_par.azimuth_freq_axis_hz.size();
    const size_t n_col = this->imaging_par.range_time_axis_sec.size();
    const size_t half_n_row = n_row / 2;
    resize_mat(this->third_comp_filt, n_row, n_col);
    std::complex<double> common_term(0.0, -4.0 * PI * square(this->imaging_par.closest_slant_range_m / this->imaging_par.sig_par.light_speed_m_s));
    for (auto i = 0; i < n_row; i++)
    {
        size_t i_sft = fft_shift_index(i, half_n_row);
        double col_term = this->modified_range_fm_rate_hz_s[i] / this->migr_par[i] * (1.0 / this->migr_par[i] - 1.0);
        for (auto j = 0; j < n_col; j++)
        {
            double row_term = 1.0;
            this->third_comp_filt[i_sft][j] = std::exp(common_term * col_term * row_term);
        }
    }
}

std::vector<std::vector<std::complex<double>>> ChirpScalingAlgo::apply_azimuth_fft(const std::vector<std::vector<std::complex<double>>> &input, bool is_ifft)
{
    assert(input.size() == this->imaging_par.azimuth_freq_axis_hz.size());
    const size_t fft_size = this->imaging_par.azimuth_freq_axis_hz.size();
    fftw_complex *input_ptr, *output_ptr;
    input_ptr = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * fft_size);
    output_ptr = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * fft_size);
    fftw_plan p;
    p = fftw_plan_dft_1d(fft_size, input_ptr, output_ptr, is_ifft ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_ESTIMATE);

    const size_t n_row = this->imaging_par.azimuth_freq_axis_hz.size();
    const size_t n_col = this->imaging_par.range_time_axis_sec.size();
    std::vector<std::vector<std::complex<double>>> output;
    resize_mat(output, n_row, n_col);
    for (auto col_idx = 0; col_idx < n_col; col_idx++)
    {
        for (auto i = 0; i < n_row; i++)
        {
            input_ptr[i][0] = input[i][col_idx].real();
            input_ptr[i][1] = input[i][col_idx].imag();
        }
        fftw_execute(p);
        for (auto i = 0; i < n_row; i++)
        {
            output[i][col_idx] = std::complex<double>(output_ptr[i][0], output_ptr[i][1]);
            if (is_ifft)
            {
                output[i][col_idx] /= static_cast<double>(fft_size);
            }
        }
    }
    fftw_destroy_plan(p);
    fftw_free(input_ptr);
    fftw_free(output_ptr);
    return output;
}

std::vector<std::vector<std::complex<double>>> ChirpScalingAlgo::apply_range_fft(const std::vector<std::vector<std::complex<double>>> &input, bool is_ifft)
{
    assert(input[0].size() == this->imaging_par.range_freq_axis_hz.size());
    const size_t fft_size = this->imaging_par.range_freq_axis_hz.size();
    fftw_complex *input_ptr, *output_ptr;
    input_ptr = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * fft_size);
    output_ptr = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * fft_size);
    fftw_plan p;
    p = fftw_plan_dft_1d(fft_size, input_ptr, output_ptr, is_ifft ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_ESTIMATE);

    const size_t n_row = this->imaging_par.azimuth_freq_axis_hz.size();
    const size_t n_col = this->imaging_par.range_freq_axis_hz.size();
    std::vector<std::vector<std::complex<double>>> output;
    resize_mat(output, n_row, n_col);
    for (auto row_idx = 0; row_idx < n_row; row_idx++)
    {
        for (auto i = 0; i < n_col; i++)
        {
            input_ptr[i][0] = input[row_idx][i].real();
            input_ptr[i][1] = input[row_idx][i].imag();
        }
        fftw_execute(p);
        for (auto i = 0; i < n_col; i++)
        {
            output[row_idx][i] = std::complex<double>(output_ptr[i][0], output_ptr[i][1]);
            if (is_ifft)
            {
                output[row_idx][i] /= static_cast<double>(fft_size);
            }
        }
    }
    fftw_destroy_plan(p);
    fftw_free(input_ptr);
    fftw_free(output_ptr);
    return output;
}

std::vector<std::vector<std::complex<double>>> ChirpScalingAlgo::apply_chirp_scaling(const std::vector<std::vector<std::complex<double>>> &input, bool is_conj)
{
    const size_t n_row = this->imaging_par.azimuth_freq_axis_hz.size();
    const size_t n_col = this->imaging_par.range_time_axis_sec.size();
    std::vector<std::vector<std::complex<double>>> output;
    resize_mat(output, n_row, n_col);
    for (auto i = 0; i < n_row; i++)
    {
        for (auto j = 0; j < n_col; j++)
        {
            std::complex<double> mult = this->chirp_scaling[i][j];
            output[i][j] = input[i][j] * (is_conj ? std::conj(mult) : mult);
        }
    }
    return output;
}

std::vector<std::vector<std::complex<double>>> ChirpScalingAlgo::apply_second_phase_func(const std::vector<std::vector<std::complex<double>>> &input, bool is_conj)
{
    const size_t n_row = this->imaging_par.azimuth_freq_axis_hz.size();
    const size_t n_col = this->imaging_par.range_time_axis_sec.size();
    std::vector<std::vector<std::complex<double>>> output;
    resize_mat(output, n_row, n_col);
    for (auto i = 0; i < n_row; i++)
    {
        for (auto j = 0; j < n_col; j++)
        {
            std::complex<double> mult = this->range_comp_filt[i][j] * this->second_comp_filt[i][j];
            output[i][j] = input[i][j] * (is_conj ? std::conj(mult) : mult);
        }
    }
    return output;
}

std::vector<std::vector<std::complex<double>>> ChirpScalingAlgo::apply_third_phase_func(const std::vector<std::vector<std::complex<double>>> &input, bool is_conj)
{
    const size_t n_row = this->imaging_par.azimuth_freq_axis_hz.size();
    const size_t n_col = this->imaging_par.range_time_axis_sec.size();
    std::vector<std::vector<std::complex<double>>> output;
    resize_mat(output, n_row, n_col);
    for (auto i = 0; i < n_row; i++)
    {
        for (auto j = 0; j < n_col; j++)
        {
            std::complex<double> mult = this->azimuth_comp_filt[i][j];
            output[i][j] = input[i][j] * (is_conj ? std::conj(mult) : mult);
        }
    }
    return output;
}