#ifndef __echo_sig_gen_par__
#define __echo_sig_gen_par__
#include "../include/util.h"

class EchoSigGenPar
{
public:
    bool azi_win_en, noise_en, coherent_scatter_en;
    size_t rng_pad_time;
    double snr_db;
    double azi_pad_time;
    EchoSigGenPar(bool azi_win_en = false, size_t rng_pad_time = 1, bool noise_en = false, double snr_db = 25.0, bool coherent_scatter_en = false, double azi_pad_time = 1.0);
};

#endif