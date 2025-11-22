#ifndef __util__
#define __util__
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <complex>
#include <chrono>
#include "json.hpp"
using json = nlohmann::json;

#ifdef USE_PARALLEL_FOR
#define OMP_FOR _Pragma("omp parallel for")
#else
#define OMP_FOR
#endif

void save_json(std::string file_name, const json &j);

json load_json(const std::string &file_name);

template <typename T>
inline T square(T x) { return x * x; }

template <typename T>
inline T cube(T x) { return x * x * x; }

template <typename T>
void resize_mat(std::vector<std::vector<T>> &mat, size_t n_row, size_t n_col)
{
    mat.resize(n_row);
    for (auto &row : mat)
    {
        row.resize(n_col);
    }
}

template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>> &mat_in)
{
    std::vector<T> vec_out;
    vec_out.reserve(mat_in.size() * mat_in[0].size());
    for (const auto &row_vec : mat_in)
    {
        vec_out.insert(vec_out.end(), row_vec.begin(), row_vec.end());
    }
    return vec_out;
}

using cpx_mat = std::vector<std::vector<std::complex<double>>>;
cpx_mat operator+(const cpx_mat &left, const cpx_mat &right);
cpx_mat operator-(const cpx_mat &left, const cpx_mat &right);

const double PI = 3.14159265358979323846;

#endif