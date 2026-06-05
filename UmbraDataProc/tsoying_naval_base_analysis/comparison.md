# tsoying_naval_base — Scene Comparison

File: `2023-04-12-13-00-32_UMBRA-04_GEC.tif`

## Overview

The Gamma distribution is fitted to **intensity** values, where intensity = amplitude².
SAR amplitude pixels are first squared before fitting, because intensity (not amplitude)
follows the Gamma distribution under the standard speckle model.

The KS statistic measures the maximum gap between the empirical and fitted CDFs (0 = perfect).
The KS p-value is omitted from this report: with large SAR images (100k+ pixels),
the p-value almost always rejects the null hypothesis even for a practically good fit,
making it an unreliable indicator at this scale. Use the KS statistic directly.

## Scenes

| **scene-A** | **scene-B** |
|---|---|
| Open sea area — no ship objects present | Harbor area — ship objects present |
| row\_start=4500, row\_len=640, col\_start=4500, col\_len=720 | row\_start=4050, row\_len=640, col\_start=4180, col\_len=720 |
| ![ scene-A slice](scene-A_slice.png) | ![scene-B slice](scene-B_slice.png) |

## Gamma Fit Comparison

| Metric | scene-A | scene-B | Reasonable |
|---|---|---|---|
| shape k | 1.1073 | 0.7467 | ~1 (single-look) to ~4 (multi-look) |
| scale θ | 1305.6268 | 3808.1505 | scene-dependent |
| sample mean / fit mean = ratio | 1445.75 / 1445.75 = 1.0000 | 2843.68 / 2843.68 = 1.0000 | ratio ≈ 1.0000 |
| sample var / fit var = ratio | 1377312.98 / 1887615.75 = 1.3705 | 30854874.81 / 10829178.19 = 0.3510 | ratio ≈ 1.0000 |
| KS stat | 0.0610 | 0.1168 | < 0.05 excellent, < 0.10 good |
