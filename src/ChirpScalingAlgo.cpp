#include "ChirpScalingAlgo.h"

ChirpScalingAlgo::ChirpScalingAlgo(const ImagingPar &imaging_par) : imaging_par(imaging_par)
{
    this->n_row = this->imaging_par.azimuth_freq_axis_hz.size();
    this->n_col = this->imaging_par.range_time_axis_sec.size();
    // auto start = std::chrono::steady_clock::now();
    gen_migr_par();
    modify_range_fm_rate_hz_s();
    gen_chirp_scaling();
    gen_range_comp_filt();
    gen_second_comp_filt();
    gen_azimuth_comp_filt();
    gen_third_comp_filt();
    // auto end = std::chrono::steady_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "Time taken for constructor: "
    //           << duration.count() / 1e6
    //           << " seconds" << std::endl;
}

void ChirpScalingAlgo::gen_migr_par()
{
    this->migr_par.resize(this->imaging_par.azimuth_freq_axis_hz.size());
    OMP_FOR
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
    OMP_FOR
    for (auto i = 0; i < this->imaging_par.azimuth_freq_axis_hz.size(); i++)
    {
        double num = this->imaging_par.sig_par.light_speed_m_s * this->imaging_par.closest_slant_range_m * square(this->imaging_par.azimuth_freq_axis_hz[i]);
        double den = 2.0 * square(this->imaging_par.sensor_speed_m_s) * cube(this->imaging_par.sig_par.carrier_freq_hz) * cube(this->migr_par[i]);
        this->modified_range_fm_rate_hz_s[i] = this->imaging_par.sig_par.range_fm_rate_hz_s / (1.0 - this->imaging_par.sig_par.range_fm_rate_hz_s * num / den);
    }
}

void ChirpScalingAlgo::gen_chirp_scaling()
{
    const size_t half_n_row = this->n_row / 2;
    this->chirp_scaling.resize(this->n_row * this->n_col);
    std::complex<double> common_term(0.0, PI);
    OMP_FOR
    for (auto i = 0; i < this->n_row; i++)
    {
        size_t i_sft = fft_shift_index(i, half_n_row);
        double first_order_col_term = this->modified_range_fm_rate_hz_s[i] * (1.0 / this->migr_par[i] - 1.0);
        double second_order_col_term = -2.0 * this->imaging_par.closest_slant_range_m / (this->imaging_par.sig_par.light_speed_m_s * this->migr_par[i]);
        for (auto j = 0; j < this->n_col; j++)
        {
            double first_order_term = first_order_col_term;
            double second_order_term = square(this->imaging_par.range_time_axis_sec[j] + second_order_col_term);
            chirp_scaling[i_sft * this->n_col + j] = std::exp(common_term * first_order_term * second_order_term);
        }
    }
}

void ChirpScalingAlgo::gen_range_comp_filt()
{
    const size_t half_n_row = this->n_row / 2, half_n_col = this->n_col / 2;
    this->range_comp_filt.resize(this->n_row * this->n_col);
    std::complex<double>
        cmmon_term(0.0, PI);
    OMP_FOR
    for (auto i = 0; i < this->n_row; i++)
    {
        size_t i_sft = fft_shift_index(i, half_n_row);
        double col_term = this->migr_par[i] / this->modified_range_fm_rate_hz_s[i];
        for (auto j = 0; j < this->n_col; j++)
        {
            size_t j_sft = fft_shift_index(j, half_n_col);
            double row_term = square(this->imaging_par.range_freq_axis_hz[j]);
            this->range_comp_filt[i_sft * this->n_col + j_sft] = std::exp(cmmon_term * col_term * row_term);
        }
    }
}

void ChirpScalingAlgo::gen_second_comp_filt()
{
    const size_t half_n_row = this->n_row / 2, half_n_col = this->n_col / 2;
    this->second_comp_filt.resize(this->n_row * this->n_col);
    std::complex<double>
        common_term(0.0, 4.0 * PI * this->imaging_par.closest_slant_range_m / this->imaging_par.sig_par.light_speed_m_s);
    OMP_FOR
    for (auto i = 0; i < this->n_row; i++)
    {
        size_t i_sft = fft_shift_index(i, half_n_row);
        double col_term = 1.0 / this->migr_par[i] - 1.0;
        for (auto j = 0; j < this->n_col; j++)
        {
            size_t j_sft = fft_shift_index(j, half_n_col);
            double row_term = this->imaging_par.range_freq_axis_hz[j];
            this->second_comp_filt[i_sft * this->n_col + j_sft] = std::exp(common_term * col_term * row_term);
        }
    }
}

void ChirpScalingAlgo::gen_azimuth_comp_filt()
{
    const size_t half_n_row = this->n_row / 2;
    this->azimuth_comp_filt.resize(this->n_row * this->n_col);
    std::complex<double> common_term(0.0, 4.0 * PI / this->imaging_par.sig_par.wavelength_m);
    OMP_FOR
    for (auto i = 0; i < this->n_row; i++)
    {
        size_t i_sft = fft_shift_index(i, half_n_row);
        double col_term = this->migr_par[i];
        for (auto j = 0; j < this->n_col; j++)
        {
            double row_term = this->imaging_par.range_time_axis_sec[j] / 2.0 * 3e8;
            this->azimuth_comp_filt[i_sft * this->n_col + j] = std::exp(common_term * col_term * row_term);
        }
    }
}

void ChirpScalingAlgo::gen_third_comp_filt()
{
    const size_t half_n_row = this->n_row / 2;
    this->third_comp_filt.resize(this->n_row * this->n_col);
    std::complex<double> common_term(0.0, -4.0 * PI * square(this->imaging_par.closest_slant_range_m / this->imaging_par.sig_par.light_speed_m_s));
    OMP_FOR
    for (auto i = 0; i < this->n_row; i++)
    {
        size_t i_sft = fft_shift_index(i, half_n_row);
        double col_term = this->modified_range_fm_rate_hz_s[i] / this->migr_par[i] * (1.0 / this->migr_par[i] - 1.0);
        for (auto j = 0; j < this->n_col; j++)
        {
            double row_term = 1.0;
            this->third_comp_filt[i_sft * this->n_col + j] = std::exp(common_term * col_term * row_term);
        }
    }
}

std::vector<std::complex<double>> ChirpScalingAlgo::apply_azimuth_fft(const std::vector<std::complex<double>> &input, bool is_ifft)
{
    const size_t fft_size = this->n_row;
    fftw_complex *input_ptr, *output_ptr;
    input_ptr = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * fft_size);
    output_ptr = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * fft_size);
    fftw_plan p;
    p = fftw_plan_dft_1d(fft_size, input_ptr, output_ptr, is_ifft ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_ESTIMATE);

    std::vector<std::complex<double>> output(this->n_row * this->n_col);
    // OMP_FOR
    for (auto col_idx = 0; col_idx < this->n_col; col_idx++)
    {
        for (auto i = 0; i < this->n_row; i++)
        {
            input_ptr[i][0] = input[i * this->n_col + col_idx].real();
            input_ptr[i][1] = input[i * this->n_col + col_idx].imag();
        }
        fftw_execute(p);
        for (auto i = 0; i < this->n_row; i++)
        {
            output[i * this->n_col + col_idx] = std::complex<double>(output_ptr[i][0], output_ptr[i][1]);
            if (is_ifft)
            {
                output[i * this->n_col + col_idx] /= static_cast<double>(fft_size);
            }
        }
    }
    fftw_destroy_plan(p);
    fftw_free(input_ptr);
    fftw_free(output_ptr);
    return output;
}

std::vector<std::complex<double>> ChirpScalingAlgo::apply_range_fft(const std::vector<std::complex<double>> &input, bool is_ifft)
{
    const size_t fft_size = this->n_col;
    fftw_complex *input_ptr, *output_ptr;
    input_ptr = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * fft_size);
    output_ptr = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * fft_size);
    fftw_plan p;
    p = fftw_plan_dft_1d(fft_size, input_ptr, output_ptr, is_ifft ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_ESTIMATE);

    std::vector<std::complex<double>> output(this->n_row * this->n_col);
    // OMP_FOR
    for (auto row_idx = 0; row_idx < this->n_row; row_idx++)
    {
        for (auto i = 0; i < this->n_col; i++)
        {
            input_ptr[i][0] = input[row_idx * this->n_col + i].real();
            input_ptr[i][1] = input[row_idx * this->n_col + i].imag();
        }
        fftw_execute(p);
        for (auto i = 0; i < this->n_col; i++)
        {
            output[row_idx * this->n_col + i] = std::complex<double>(output_ptr[i][0], output_ptr[i][1]);
            if (is_ifft)
            {
                output[row_idx * this->n_col + i] /= static_cast<double>(fft_size);
            }
        }
    }
    fftw_destroy_plan(p);
    fftw_free(input_ptr);
    fftw_free(output_ptr);
    return output;
}

std::vector<std::complex<double>> ChirpScalingAlgo::apply_chirp_scaling(const std::vector<std::complex<double>> &input, bool is_conj)
{
    std::vector<std::complex<double>> output(this->n_row * this->n_col);
    OMP_FOR
    for (auto i = 0; i < this->n_row; i++)
    {
        for (auto j = 0; j < this->n_col; j++)
        {
            std::complex<double> mult = this->chirp_scaling[i * this->n_col + j];
            output[i * this->n_col + j] = input[i * this->n_col + j] * (is_conj ? std::conj(mult) : mult);
        }
    }
    return output;
}

std::vector<std::complex<double>> ChirpScalingAlgo::apply_second_phase_func(const std::vector<std::complex<double>> &input, bool is_conj)
{
    std::vector<std::complex<double>> output(this->n_row * this->n_col);
    OMP_FOR
    for (auto i = 0; i < this->n_row; i++)
    {
        for (auto j = 0; j < this->n_col; j++)
        {
            std::complex<double> mult = this->range_comp_filt[i * this->n_col + j] * this->second_comp_filt[i * this->n_col + j];
            output[i * this->n_col + j] = input[i * this->n_col + j] * (is_conj ? std::conj(mult) : mult);
        }
    }
    return output;
}

std::vector<std::complex<double>> ChirpScalingAlgo::apply_third_phase_func(const std::vector<std::complex<double>> &input, bool is_conj)
{
    std::vector<std::complex<double>> output(this->n_row * this->n_col);
    OMP_FOR
    for (auto i = 0; i < this->n_row; i++)
    {
        for (auto j = 0; j < this->n_col; j++)
        {
            std::complex<double> mult = this->azimuth_comp_filt[i * this->n_col + j];
            output[i * this->n_col + j] = input[i * this->n_col + j] * (is_conj ? std::conj(mult) : mult);
        }
    }
    return output;
}

std::vector<std::complex<double>> ChirpScalingAlgo::apply_csa(const std::vector<std::complex<double>> &input)
{
    std::vector<std::complex<double>> azimuth_fft_out = this->apply_azimuth_fft(input);
    std::vector<std::complex<double>> chirp_scaling_out = this->apply_chirp_scaling(azimuth_fft_out);
    std::vector<std::complex<double>> range_fft_out = this->apply_range_fft(chirp_scaling_out);
    std::vector<std::complex<double>> second_phase_func_out_out = this->apply_second_phase_func(range_fft_out);
    std::vector<std::complex<double>> range_ifft_out = this->apply_range_fft(second_phase_func_out_out, true);
    std::vector<std::complex<double>> third_phase_func_out = this->apply_third_phase_func(range_ifft_out);
    std::vector<std::complex<double>> output = this->apply_azimuth_fft(third_phase_func_out, true);
    return output;
}

std::vector<std::complex<double>> ChirpScalingAlgo::apply_inverse_csa(const std::vector<std::complex<double>> &input)
{
    std::vector<std::complex<double>> azimuth_fft_out = this->apply_azimuth_fft(input);
    std::vector<std::complex<double>> third_phase_func_out = this->apply_third_phase_func(azimuth_fft_out, true);
    std::vector<std::complex<double>> range_fft_out = this->apply_range_fft(third_phase_func_out);
    std::vector<std::complex<double>> second_phase_func_out_out = this->apply_second_phase_func(range_fft_out, true);
    std::vector<std::complex<double>> range_ifft_out = this->apply_range_fft(second_phase_func_out_out, true);
    std::vector<std::complex<double>> chirp_scaling_out = this->apply_chirp_scaling(range_ifft_out, true);
    std::vector<std::complex<double>> output = this->apply_azimuth_fft(chirp_scaling_out, true);
    return output;
}