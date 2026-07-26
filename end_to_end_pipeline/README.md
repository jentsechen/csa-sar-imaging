# End-to-End Pipeline

Validates whether the CSA imaging pipeline preserves ship-detection accuracy:
takes real HRSID JPGs, runs them through point-target extraction -> echo
signal simulation -> CSA focusing -> magnitude/dB -> JPG, then compares YOLO
detection metrics against the original images. Corresponds to milestone
issue #35 ("[8.1] End-to-end pipeline script").

## Pipeline

```
images/<stem>.jpg
  -> [jpg_to_point_target.py]      -> point_target_location/<stem>.json
  -> [gen_echo_signal_batch.py]    -> echo_signal/<stem>.npy
  -> [csa_to_jpg_batch.py]
       TestMultiPointTarget focus    -> focused_image/<stem>.npy
       TestMultiPointTarget calc_mag -> focused_image/<stem>_mag_db.npy
       crop + 30dB clip + normalize  -> csa_jpg/<stem>.jpg
  -> [eval_csa_jpg.py] / [compare_original_vs_csa.py] -> Precision/Recall/mAP@0.5
```

## Scripts (run in order)

| Script | Input | Output | Notes |
|---|---|---|---|
| `select_images.py` | `sar_ship_detect/HRSID_YOLO/images/val` | `images/`, `labels/`, `selected_images.txt` | Random sample (seed=0, reproducible), copies JPGs + matching GT labels |
| `jpg_to_point_target.py` | `images/*.jpg` | `point_target_location/*.json` | Grayscale + threshold (currently 150); `METHODS` dict designed for adding other methods later |
| `gen_echo_signal_batch.py --n N` | `point_target_location/*.json`, `input_par.json` | `echo_signal/*.npy` | Wraps `../build/gen_echo_signal`; `--n` controls how many scenes to process (for timing) |
| `csa_to_jpg_batch.py --n N` | `echo_signal/*.npy` | `focused_image/*.npy`, `focused_image/*_mag_db.npy`, `csa_jpg/*.jpg` | Wraps `../build/TestMultiPointTarget focus` + `calc_mag`; crops center 800x800, 30dB dynamic range |
| `eval_csa_jpg.py` | `csa_jpg/*.jpg`, `labels/*.txt` | `csa_eval/` (YOLO val run) | Precision/Recall/mAP@0.5 on CSA-reconstructed images only |
| `compare_original_vs_csa.py` | `images/`, `csa_jpg/`, `labels/` | `original_eval/`, `csa_eval/` | Same metrics for **original** vs **csa**, over the same scene set, side by side |

`gen_echo_signal_batch.py` and `csa_to_jpg_batch.py` only process scenes that
have completed the previous stage, so partial runs (`--n 10`) are safe to
extend later by rerunning with a larger `--n`.

## Directories

- `images/`, `labels/` — the 100 selected source JPGs + ground-truth YOLO labels
- `point_target_location/` — thresholded point-target JSON per image
- `echo_signal/` — simulated raw SAR echo (complex `.npy`, one per processed scene)
- `focused_image/` — CSA output (`.npy`) and dB magnitude (`_mag_db.npy`)
- `csa_jpg/` — final grayscale JPG reconstructed from CSA output, ready for `infer.py`
- `original_eval/`, `csa_eval/` — temp YOLO eval sets (images+labels symlinks, `data.yaml`) built by the comparison scripts; safe to delete and regenerate

## Status

100 images selected; **10/100** have been run through the full pipeline so far
(`--n 10`). Latest comparison (10 scenes, 22 GT instances):

| Set | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| original | 1.000 | 0.955 | 0.955 | 0.806 |
| csa | 0.984 | 1.000 | 0.995 | 0.453 |

**Caveat:** 10 scenes is far below the ~500-800 image threshold found in
[`sar_ship_detect/docs/sample_size_validation.md`](../sar_ship_detect/docs/sample_size_validation.md)
for stable Precision/Recall/mAP estimates — treat these numbers as a pipeline
smoke test, not a conclusive result. Extend with `--n 100` (or beyond, by
selecting more images) once timing budget allows; per-scene timing so far:
~33 s/scene for echo signal generation (the bottleneck) and ~2 s/scene for
CSA + dB + JPG conversion.

## Configuration

- `input_par.json` — HRSID sensor parameters (same as used for
  `P0002_1800_2600_2400_3200` in `TestMultiPointTarget/TestMultiPointTarget.py`),
  shared across all scenes since they come from the same sensor/dataset.
- YOLO weights: `sar_ship_detect/weights/best.pt`, conf=0.25, iou=0.45, imgsz=800
  (matching `sar_ship_detect/infer.py` defaults).
