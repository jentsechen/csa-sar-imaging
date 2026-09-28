# End-to-End Pipeline (Union-Mask Method)

Validates whether the CSA imaging pipeline preserves ship-detection accuracy,
using a GT/prediction bounding-box union mask (instead of intensity
thresholding) to isolate detection-relevant pixels before echo simulation +
CSA reconstruction. Corresponds to milestone issue #35 ("[8.1] End-to-end
pipeline script").

## Pipeline

The echo + imaging stages now run on the GPU via [`sarsim`](../sarsim/README.md)
(recommended); the original C++ wrappers are kept below as the legacy path.

```
images/<stem>.jpg  (1593 scenes, HRSID val)
  -> [mask_outside_gt_pred_union.py]
       run inference on original_eval/images, zero out every pixel outside
       the union of (GT boxes) and (predicted boxes)
       -> union_masked/images/<stem>.jpg                  (detector input, lossy)
       -> union_pipeline/point_target_location/<stem>.png (same array, lossless:
          echo-simulation input; --json also writes <stem>.json for the C++ path)
       -> union_masked/nonzero_pct.csv  (% nonzero pixels kept per image)

  GPU path (python -m sarsim.pipeline --run <run> --algo csa):
       point_target_location/<stem>.png -> echo (CUDA kernel, stays on GPU)
       -> CSA (CuPy/cuFFT) -> crop + |x|^2 + 30dB clip
       -> union_pipeline/runs/<run>/csa/jpg/<stem>.jpg
       -> union_pipeline/runs/<run>/csa/power/<stem>.npy  (float32 linear power)

  Legacy C++ path:
  -> [gen_echo_signal_union_batch.py]
       point_target_location/<stem>.png -> <stem>.json (if not already written)
       (no additional intensity threshold -- the box union IS the mask)
       -> union_pipeline/echo_signal/<stem>.npy
       (azi_win_en=False; runtime capped via --max-seconds, predicted from
       nonzero pixel count at ~0.0462 s/px)

  -> [csa_to_jpg_union_batch.py]
       TestMultiPointTarget focus    -> union_pipeline/focused_image/<stem>.npy
       TestMultiPointTarget calc_mag -> union_pipeline/focused_image/<stem>_mag_db.npy
       crop + 30dB clip + normalize  -> union_pipeline/csa_jpg/<stem>.jpg

  -> [eval_union_masked.py] / [eval_union_csa.py] -> Precision/Recall/mAP@0.5
```

## Scripts (run in order)

| Script | Input | Output | Notes |
|---|---|---|---|
| `select_images.py` | `sar_ship_detect/HRSID_YOLO/images/val` | `images/`, `labels/`, `selected_images.txt` | Random sample (seed=0), `--region offshore` restricts to the official HRSID offshore split |
| `mask_outside_gt_pred_union.py` | `original_eval/images`, `original_eval/labels` | `union_masked/images/*.jpg`, `union_pipeline/point_target_location/*.png` (`--json`: also `*.json`), `union_masked/nonzero_pct.csv` | Runs inference once on the original images; zeroes pixels outside GT∪Pred box union and writes the result both as JPG (for YOLO) and lossless PNG (point targets). Point targets must not come from the JPG: its ringing adds ~11% spurious low-amplitude scatterers outside the boxes |
| `eval_union_masked.py` | `union_masked/images` | `union_masked_eval/` (YOLO val run) | Sanity check: confirms masking alone doesn't change detection metrics |
| `gen_echo_signal_union_batch.py --max-seconds N` (legacy) | `union_pipeline/point_target_location/*.png` | `union_pipeline/point_target_location/*.json`, `union_pipeline/echo_signal/*.npy`, `union_pipeline/echo_signal_timing.csv`, `union_pipeline/skipped_scenes.txt` | Wraps `../build/gen_echo_signal`; writes its own `input_par.json` with `azi_win_en=False`; scenes whose predicted runtime exceeds `--max-seconds` are skipped |
| `csa_to_jpg_union_batch.py --n N` (legacy) | `union_pipeline/echo_signal/*.npy` | `union_pipeline/focused_image/*.npy`, `union_pipeline/focused_image/*_mag_db.npy`, `union_pipeline/csa_jpg/*.jpg` | Wraps `../build/TestMultiPointTarget focus` + `calc_mag`; crops center 800x800, 30dB dynamic range |
| `python -m sarsim.pipeline --run R --algo csa` (from repo root; recommended) | `union_pipeline/point_target_location/*.png` | `union_pipeline/runs/R/{csa/jpg,csa/power,metrics.csv,manifest.json}` | Echo CUDA kernel -> CuPy CSA -> JPG on the GPU, no echo/focused `.npy` written; ~4 min for all 1593 scenes. Several `--algo` share each echo. See `../sarsim/README.md` |
| `eval_union_csa.py [--csa-dir DIR --name NAME]` | `images/`, `DIR` (default `union_pipeline/csa_jpg/`), `labels/` | `original_union_eval/`, `NAME_union_eval/` | Precision/Recall/mAP@0.5, original vs union-mask-CSA, over exactly the stems present in `DIR` |

`gen_echo_signal_union_batch.py`, `csa_to_jpg_union_batch.py` and
`sarsim.pipeline` only process scenes not yet done, so interrupted runs (e.g.
after a disconnect) resume without recomputation.

**Faulty CPU:** every segfault on this workstation happens on CPU 1 (core 4);
`sarsim` avoids it automatically, other long jobs should run under
`taskset -c 0,2-23`. See `../gpu_prototype/README.md`.

## Directories

- `images/`, `labels/` — the selected source JPGs + ground-truth YOLO labels
- `original_eval/` — the same images repackaged as a YOLO eval set (images+labels symlinks); source for the union mask
- `union_masked/` — masked JPGs + per-image nonzero-pixel-percentage CSV
- `union_masked_eval/` — temp YOLO eval set for `union_masked/images`
- `union_pipeline/` — working dir for the echo/imaging stages:
  - `point_target_location/` (`<stem>.png`, lossless masked image; ~12 MB total), `input_par.json` (azi_win_en=False) — shared input to both paths
  - `runs/<run>/` — `sarsim.pipeline` outputs (JPG, float32 power, metrics, manifest)
  - `reference/` — two scenes' C++ point-target JSON (JPEG-derived, as used then) + echo + focused `.npy`, used by `python -m sarsim.validate`
  - `csa_jpg/` — legacy C++ JPGs (1583 scenes; `P0062_3500_4300_1800_2600` is corrupt, see below)
  - `echo_signal/`, `focused_image/` — legacy C++ intermediates, 164 MB/scene; deleted 2026-09-28 (~670 GB), recreated only if the legacy scripts are rerun
- `*_union_eval/` — temp YOLO eval sets built by `eval_union_csa*.py` (git-ignored)

## Status

### GPU pipeline result (2026-09-28, 1593 scenes, 3081 GT instances)

`python -m sarsim.pipeline --run union_png --algo csa`, then
`eval_union_csa.py --csa-dir union_pipeline/runs/union_png/csa/jpg --name union_png`:

| Set | Precision | Recall | mAP@0.5 |
|---|---|---|---|
| original | 0.9818 | 0.9651 | 0.9838 |
| **union_png** (GPU, lossless PNG point targets, azi_win_en=False) | 0.9737 | 0.9747 | 0.9838 |
| baseline_csa (GPU, point targets read back from the masked JPG) | 0.9741 | 0.9753 | 0.9843 |

`baseline_csa` built its point targets from `union_masked/images/*.jpg`, whose
JPEG ringing added 606,144 spurious scatterers outside the boxes (11% of
5,502,028; values 1-9, mean 1.6). Removing them (`union_png`, 4,886,935
targets) changes detection metrics by <=0.0006 and cuts echo time by 11%, so
the ringing did not affect the earlier conclusions; `union_png` is the
reference run from now on.

Legacy C++ `csa_jpg/` on its 1583 scenes, for comparison:

| Set | Precision | Recall | mAP@0.5 |
|---|---|---|---|
| original | 0.9841 | 0.9663 | 0.9844 |
| union_csa (C++) | 0.9740 | 0.9756 | 0.9844 |

The GPU and C++ pipelines give the same detection performance, and CSA
reconstruction of the union-mask scenes no longer costs mAP@0.5: recall rises
~1 pt and precision drops ~0.8 pt relative to the original images. The C++
set includes one corrupt image (`P0062_3500_4300_1800_2600`, flat focused
output — see `../gpu_prototype/README.md`) and lacks the 10 scenes that were
skipped or failed there.

## Timing

- GPU (`sarsim.pipeline`, RTX 5090, 1593 scenes, `union_png`): 3.8 min end to
  end -- echo 197 s total (~0.12 s/scene, ~40 ms per 1000 targets), CSA +
  post-processing 24 s total (~15 ms/scene, independent of target count).

## Configuration

- `union_pipeline/input_par.json` — HRSID sensor parameters (same as used
  elsewhere in this repo for `P0002_1800_2600_2400_3200`), with
  **`azi_win_en: false`**.
- YOLO weights: `sar_ship_detect/weights/best.pt`, conf=0.25, iou=0.45,
  imgsz=800 (matching `sar_ship_detect/infer.py` defaults).

## Post-CSA Threshold Analysis

Follow-up experiment on top of the union-mask pipeline above: does applying
a fixed intensity threshold to the *CSA output* (zeroing pixels below T,
separate from the union-mask step) clean up residual reconstruction noise
without damaging real ship signal? All figures/data from this analysis are
saved to `../diagram/thresholding/union_csa/` (repo-level `diagram/` folder,
not under `end_to_end_pipeline/`).

| Script | Input | Output | Notes |
|---|---|---|---|
| `threshold_union_csa.py --threshold T` | `union_pipeline/csa_jpg/*.jpg` | `union_pipeline/csa_jpg_t<T>/*.jpg` | Thresholds all completed scenes; `T` tested so far: 80, 100, 120 |
| `eval_union_csa_threshold.py --threshold T` | `images/`, `union_pipeline/csa_jpg/`, `union_pipeline/csa_jpg_t<T>/` | `original_union_eval/`, `union_csa_union_eval/`, `union_csa_t<T>_union_eval/` | Precision/Recall/mAP@0.5, three-way: original vs union_csa (no threshold) vs union_csa_t\<T\> |
| `find_threshold_diff_images.py --threshold T` | same as above | `../diagram/thresholding/union_csa/per_image_diff_t<T>.json` | Per-image GT/prediction IoU matching (TP/FP/FN) for all three sets; prints scenes where thresholding changes the detection outcome, and flags candidates where it recovers toward original without exceeding it |

### Cross-section plotting (`crosssection_plots/`)

Scripts that visualize a specific scene's row/column intensity profile
and/or the raw images themselves, to see *why* thresholding helps or hurts
a given detection. Run from inside `crosssection_plots/` (or adjust
relative paths accordingly):

```bash
cd crosssection_plots

# Line-plot cross-sections (union_masked input vs CSA output vs CSA+threshold),
# saved as two SEPARATE files (_horizontal.png / _vertical.png). By default
# also overlays GT box edges (tab:purple) and the model's predicted box
# edges on the original image (tab:cyan) wherever a box crosses the cut.
python plot_crosssection_scene.py <stem> \
    --row <R> --col <C> \
    --row-window <R0> <R1> --col-window <C0> <C1> \
    --threshold 120
#   --no-boxes              disable the GT/detect box overlay
#   --thresholded-path P    read the CSA+threshold image from P instead of
#                           union_pipeline/csa_jpg_t<T>/<stem>.jpg -- use a
#                           lossless .png here to avoid small-value ripple
#                           from re-encoding a thresholded array as JPEG
#   --out PATH.png          override the output path (still split into
#                           PATH_horizontal.png / PATH_vertical.png)
#   --normalize             scale each curve to its own peak (=1.0) within
#                           the plotted window -- OBSERVATION ONLY. union_masked
#                           (raw pixel value) and CSA output (independently
#                           re-normalized per-image to its own dB peak, see
#                           csa_to_jpg_union_batch.py) are not on a shared
#                           absolute intensity scale, so raw 0-255 values are
#                           not directly comparable across the three curves --
#                           this flag makes relative shape/magnitude comparable
#                           instead. Without it, y-axis stays raw 0-255.

# Same cut row/col, but shown directly on the cropped images themselves
# (three separate .png files: _union_masked / _csa / _csa_t<T>), with a red
# line marking the cut and GT (tab:purple) / detect (tab:cyan) box rectangles.
python plot_crosssection_overlay_on_images.py <stem> \
    --row <R> --col <C> \
    --row-window <R0> <R1> --col-window <C0> <C1> \
    --threshold 120

# One-off, hardcoded to P0002_3600_4400_1200_2000 (row=75, col=699) -- the
# original ship-crop cross-section from before plot_crosssection_scene.py
# was generalized.
python plot_point_target_vs_csa_crosssection.py --threshold 120
```

Both `plot_crosssection_scene.py` and `plot_crosssection_overlay_on_images.py`
re-run YOLO on `images/<stem>.jpg` each time to get the "detect box" (GT
boxes come from `labels/<stem>.txt`) -- this is the same box source
`mask_outside_gt_pred_union.py` unions to build `union_masked/`, just kept
as two separate sets here instead of merged, so GT vs. detection can be
compared directly.
