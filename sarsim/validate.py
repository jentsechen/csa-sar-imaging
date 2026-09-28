#!/usr/bin/env python3
"""Regression check of the GPU echo kernel and imaging algorithms against the
C++ reference outputs kept in end_to_end_pipeline/union_pipeline/reference/
(point_target_location/<stem>.json input, echo_signal/<stem>.npy from
gen_echo_signal, focused_image/<stem>.npy from TestMultiPointTarget focus --
all from the original C++ pipeline).

Usage:
    python -m sarsim.validate
"""
import argparse
import json
import os
import time

import numpy as np

from .pipeline import UNION, exclude_cpus

REF = os.path.join(UNION, "reference")
STEMS = ["P0033_1800_2600_4200_5000", "P0048_2900_3700_3600_4400"]
ECHO_RTOL = 1e-7    # kernel vs C++: float-order noise, ~5e-9 observed
FOCUS_RTOL = 1e-12  # CSA on the same C++ echo: ~5e-16 observed


def rel_err(a, ref):
    return float(np.max(np.abs(a - ref)) / np.max(np.abs(ref)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*", default=STEMS)
    args = ap.parse_args()
    exclude_cpus("1")

    import cupy as cp
    from .echo import gen_echo_signal
    from .imaging.csa import ChirpScalingAlgo
    from .params import build_imaging_axes, build_point_target_list, load_input_par

    par = load_input_par(os.path.join(UNION, "input_par.json"))
    ax = build_imaging_axes(par)
    csa = ChirpScalingAlgo(cp, ax, par)
    ok = True
    for stem in args.stems:
        # The reference echoes were generated from these (JPEG-derived) JSON
        # masks, so they stay the reference input even though the pipeline
        # now uses lossless PNG masks.
        with open(os.path.join(REF, "point_target_location", stem + ".json")) as f:
            mask = json.load(f)
        a, r, c = build_point_target_list(mask, ax["n_row"], ax["n_col"],
                                          ax["pulse_rep_freq_hz"], ax["sampling_freq_hz"])
        ref_echo = np.load(os.path.join(REF, "echo_signal", stem + ".npy"))
        t0 = time.perf_counter()
        echo = cp.asnumpy(gen_echo_signal(ax, a, r, c)[0])
        t_echo = time.perf_counter() - t0
        e_echo = rel_err(echo, ref_echo)

        ref_focus = np.load(os.path.join(REF, "focused_image", stem + ".npy"))
        focus = cp.asnumpy(csa.focus(cp.asarray(ref_echo)))
        e_focus = rel_err(focus, ref_focus)

        passed = e_echo < ECHO_RTOL and e_focus < FOCUS_RTOL
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'} {stem} ({len(a)} targets): "
              f"echo rel err {e_echo:.2e} ({t_echo:.2f}s), csa rel err {e_focus:.2e}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
