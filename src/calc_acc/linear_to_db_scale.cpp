#include "../../include/util.h"
#include "../../include/util.cpp"

int main(int argc, char *argv[])
{
    cnpy::NpyArray arr = cnpy::npy_load(std::string(argv[1]) + ".npy");
    const std::vector<size_t> &shape_vec = arr.shape;
    assert(shape_vec.size() == 2);
    size_t n_row = shape_vec[0], n_col = shape_vec[1];
    std::complex<double> *ptr = arr.data<std::complex<double>>();
    std::vector<std::complex<double>> focused_image(ptr, ptr + n_row * n_col);
    std::vector<double> focused_image_mag_db(n_row * n_col);
    OMP_FOR
    for (auto i = 0; i < n_row; i++)
    {
        for (auto j = 0; j < n_col; j++)
        {
            focused_image_mag_db[i * n_col + j] = 10.0 * std::log10(square(focused_image[i * n_col + j].real()) + square(focused_image[i * n_col + j].imag()));
        }
    }
    save_mat_to_npy(std::string(argv[1]) + "_mag_db.npy", focused_image_mag_db, n_row, n_col);
}