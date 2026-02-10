#include "EchoSigGenPar.h"

EchoSigGenPar::EchoSigGenPar(bool azi_win_en, size_t rng_pad_time, bool noise_en, double snr_db)
    : azi_win_en(azi_win_en), rng_pad_time(rng_pad_time), noise_en(noise_en), snr_db(snr_db) {}