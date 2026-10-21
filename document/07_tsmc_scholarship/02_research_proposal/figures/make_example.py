"""Figure 1 of the proposal: CSA and L1/2+TV images of P0033 at SNR -5 dB (noise seed 0,
per-level parameters of end_to_end_pipeline/noise/full), zoomed on ship 1, without boxes.

Run from end_to_end_pipeline/ on the GPU:
    taskset -c 0,2-6,8-23 python3 ../document/07_tsmc_scholarship/02_research_proposal/figures/make_example.py
"""
import argparse, sys, os
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", ".."))
import cupy as cp
from sarsim.noise_pipeline import Context, clean_crop, scene_power, noise_field, noisy, load_params, snr_name, MARGIN
from sarsim.speckle_pipeline import to_gray
from sarsim.refine import apply

stem, snr, out_dir = "P0033_1800_2600_4200_5000", -5.0, HERE
x0, y0, x1, y1 = 650, 83, 778, 211
params_c, _ = load_params(argparse.Namespace(run="full", seed=0))
ctx = Context(cp)
crop = clean_crop(ctx, ctx.load_amp(stem))
p = scene_power(cp, crop)
w = cp.asarray(noise_field(stem, crop.shape, 0))
for m in ("csa", "l12tv"):
    img, _ = apply(cp, m, noisy(cp, crop, w, snr, p), params_c[snr_name(snr)][m], MARGIN)
    g = to_gray(cp, img)
    roi = g[y0:y1, x0:x1]
    cv2.imwrite(os.path.join(out_dir, f"example_{'csa' if m == 'csa' else 'l12tv'}_snr-5.png"), cv2.resize(roi, (512, 512), interpolation=cv2.INTER_NEAREST))
    print(m, params_c[snr_name(snr)][m])
