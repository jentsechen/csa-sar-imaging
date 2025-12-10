#include "util.h"

void save_json(std::string file_name, const json &j)
{
    std::ofstream output_file(file_name);
    if (output_file.is_open())
    {
        output_file << j.dump(4);
        output_file.close();
        std::cout << "Successfully saved JSON to " << file_name << std::endl;
    }
    else
    {
        std::cerr << "Error: Unable to open file " << file_name << " for writing." << std::endl;
    }
}

json load_json(const std::string &file_name)
{
    std::ifstream input_file(file_name);
    if (!input_file.is_open())
    {
        std::cerr << "Error: Could not open file " << file_name << std::endl;
        return nullptr;
    }
    try
    {
        json data = json::parse(input_file);
        std::cout << "Successfully parsed JSON from " << file_name << std::endl;
        return data;
    }
    catch (const json::parse_error &e)
    {
        std::cerr << "Error: JSON parsing failed in " << file_name << ": " << e.what() << std::endl;
        return nullptr;
    }
}

cpx_mat operator+(const cpx_mat &left, const cpx_mat &right)
{
    assert(left.size() == right.size());
    size_t n_row = left.size(), n_col = left[0].size();
    cpx_mat result(n_row);
    OMP_FOR
    for (auto i = 0; i < n_row; i++)
    {
        result[i].resize(n_col);
        assert(left[i].size() == right[i].size());
        for (auto j = 0; j < n_col; j++)
        {
            result[i][j] = left[i][j] + right[i][j];
        }
    }
    return result;
}

cpx_mat operator-(const cpx_mat &left, const cpx_mat &right)
{
    assert(left.size() == right.size());
    size_t n_row = left.size(), n_col = left[0].size();
    cpx_mat result(n_row);
    OMP_FOR
    for (auto i = 0; i < n_row; i++)
    {
        result[i].resize(n_col);
        assert(left[i].size() == right[i].size());
        for (auto j = 0; j < n_col; j++)
        {
            result[i][j] = left[i][j] - right[i][j];
        }
    }
    return result;
}

std::vector<std::complex<double>> operator+(const std::vector<std::complex<double>> &left, const std::vector<std::complex<double>> &right)
{
    assert(left.size() == right.size());
    size_t n_total = left.size();
    std::vector<std::complex<double>> result(n_total);
    OMP_FOR
    for (auto i = 0; i < n_total; i++)
    {
        result[i] = left[i] + right[i];
    }
    return result;
}

std::vector<std::complex<double>> operator-(const std::vector<std::complex<double>> &left, const std::vector<std::complex<double>> &right)
{
    assert(left.size() == right.size());
    size_t n_total = left.size();
    std::vector<std::complex<double>> result(n_total);
    OMP_FOR
    for (auto i = 0; i < n_total; i++)
    {
        result[i] = left[i] - right[i];
    }
    return result;
}

// void save_mat_to_npy(std::string file_path, const std::vector<std::complex<double>> &mat, size_t n_row, size_t n_col)
// {
//     cnpy::npy_save(file_path, mat.data(), {n_row, n_col}, "w");
// }