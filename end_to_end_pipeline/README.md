# End-to-End Pipeline (Union-Mask Method)

Validates whether the CSA imaging pipeline preserves ship-detection accuracy,
using a GT/prediction bounding-box union mask (instead of intensity
thresholding) to isolate detection-relevant pixels before echo simulation +
CSA reconstruction. Corresponds to milestone issue #35 ("[8.1] End-to-end
pipeline script").

## Pipeline

```
images/<stem>.jpg  (100 offshore scenes, HRSID val)
  -> [mask_outside_gt_pred_union.py]
       run inference on original_eval/images, zero out every pixel outside
       the union of (GT boxes) and (predicted boxes)
       -> union_masked/images/<stem>.jpg
       -> union_masked/nonzero_pct.csv  (% nonzero pixels kept per image)

  -> [gen_echo_signal_union_batch.py]
       union_masked jpg -> union_pipeline/point_target_location/<stem>.json
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
| `mask_outside_gt_pred_union.py` | `original_eval/images`, `original_eval/labels` | `union_masked/images/*.jpg`, `union_masked/nonzero_pct.csv` | Runs inference once on the original images; zeroes pixels outside GT∪Pred box union |
| `eval_union_masked.py` | `union_masked/images` | `union_masked_eval/` (YOLO val run) | Sanity check: confirms masking alone doesn't change detection metrics |
| `gen_echo_signal_union_batch.py --max-seconds N` | `union_masked/images` | `union_pipeline/point_target_location/*.json`, `union_pipeline/echo_signal/*.npy`, `union_pipeline/echo_signal_timing.csv`, `union_pipeline/skipped_scenes.txt` | Wraps `../build/gen_echo_signal`; writes its own `input_par.json` with `azi_win_en=False`; scenes whose predicted runtime exceeds `--max-seconds` are skipped |
| `csa_to_jpg_union_batch.py --n N` | `union_pipeline/echo_signal/*.npy` | `union_pipeline/focused_image/*.npy`, `union_pipeline/focused_image/*_mag_db.npy`, `union_pipeline/csa_jpg/*.jpg` | Wraps `../build/TestMultiPointTarget focus` + `calc_mag`; crops center 800x800, 30dB dynamic range |
| `eval_union_csa.py` | `images/`, `union_pipeline/csa_jpg/`, `labels/` | `original_union_eval/`, `union_csa_union_eval/` | Precision/Recall/mAP@0.5, original vs union-mask-CSA, over the same scene set |

`gen_echo_signal_union_batch.py` and `csa_to_jpg_union_batch.py` only process
scenes that completed the previous stage and skip ones already done, so
interrupted runs (e.g. after a disconnect) resume without recomputation.

## Directories

- `images/`, `labels/` — the 100 selected source JPGs + ground-truth YOLO labels (all offshore)
- `original_eval/` — the same 100 images repackaged as a YOLO eval set (images+labels symlinks); source for the union mask
- `union_masked/` — masked JPGs + per-image nonzero-pixel-percentage CSV
- `union_masked_eval/` — temp YOLO eval set for `union_masked/images`
- `union_pipeline/` — self-contained working dir for the echo/CSA stages: `point_target_location/`, `echo_signal/`, `focused_image/`, `csa_jpg/`, its own `input_par.json` (azi_win_en=False), `echo_signal_timing.csv`, `skipped_scenes.txt`
- `original_union_eval/`, `union_csa_union_eval/` — temp YOLO eval sets built by `eval_union_csa.py`

## Status

100 images selected (offshore only); **92/100** completed the full union-mask
pipeline (8 skipped — scenes whose union-mask region was too dense to finish
within the time cap; see `union_pipeline/skipped_scenes.txt`).

### Masking sanity check (100 images, no echo/CSA)

| Set | Precision | Recall | mAP@0.5 |
|---|---|---|---|
| original | 1.0000 | 0.9820 | 0.9850 |
| union_masked | 1.0000 | 0.9880 | 0.9850 |

Confirms masking alone (zeroing pixels outside the GT∪Pred box union, avg
0.41% pixels retained) does not degrade detection — a few borderline
IoU=0.5 flips occur (see conversation history / notes below) but net effect
is neutral to slightly positive.

### Full pipeline result (92 images, 138 GT instances)

| Set | Precision | Recall | mAP@0.5 |
|---|---|---|---|
| original | 1.0000 | 0.9783 | 0.9750 |
| union_csa (azi_win_en=False) | 0.9774 | 0.9399 | 0.9538 |

This is a smaller drop than the earlier intensity-threshold-based CSA
experiment (which had shown Precision 0.871 / Recall 0.822 / mAP@0.5 0.847
against the same kind of original baseline) — the union-mask approach
preserves more detection-relevant information into the CSA reconstruction
than a fixed brightness threshold does.

**Open question:** the improvement could be due to the union-mask strategy,
`azi_win_en=False`, or both — no controlled experiment has isolated the two
factors yet (would need a union-mask run with `azi_win_en=True` for
comparison).

## Timing

- Echo signal generation (union mask, 92 scenes): ~2.86 hours total, runtime
  scales linearly with nonzero pixel count (~0.0462 s/px, r=0.9999)
- CSA focus + dB + JPG conversion (92 scenes): 201.4 s total (~2.19 s/scene)
- 8 scenes skipped at `--max-seconds 300` (predicted 301–767s); see
  `union_pipeline/skipped_scenes.txt` for the list

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

### Findings so far (92 scenes)

| Set | Precision | Recall | mAP@0.5 |
|---|---|---|---|
| original | 1.0000 | 0.9783 | 0.9750 |
| union_csa (no threshold) | 0.9774 | 0.9399 | 0.9538 |
| union_csa_t80 | 0.9847 | 0.9855 | 0.9841 |
| union_csa_t100 | 0.9918 | 0.9855 | 0.9844 |
| union_csa_t120 | 0.9927 | 0.9855 | 0.9847 |

Thresholding the CSA output (on top of the union mask) recovers most of the
gap to `original` -- Recall saturates at 0.9855 from T=80 upward, while
Precision/mAP keep improving only marginally as T increases to 120. Isolated
per-scene inspection (`find_threshold_diff_images.py` +
`crosssection_plots/`) shows two distinct mechanisms behind this recovery:
thresholding can either clean up an unrelated false-positive elsewhere in
the scene, or directly narrow a ship's own CSA-blurred signal enough to push
its detection IoU back above 0.5 (e.g. `P0070_600_1400_3600_4400`, where
narrowing the box from IoU=0.44 to IoU=0.71 flips a miss into a match).
Pushing the threshold higher than ~120 (tested up to 150 on a single crop)
does not converge to a clean ship -- it keeps eroding real signal at the
same rate it removes residual background, since the two overlap in
intensity.
