# Validation Sample-Size Study

Question: instead of running `model.val()` over all 1,962 HRSID val images, how
few images are needed to get a Precision / Recall / mAP@0.5 estimate that
reliably matches the full-set result?

## Method

**Analytical estimate.** Precision and Recall are proportions, so the standard
sample-size formula for estimating a proportion applies:

```
n = z^2 * p(1-p) / e^2
```

With a 95% confidence level (z = 1.96) and p ≈ 0.85 (typical Recall), this
gives roughly:

| Margin of error | Instances needed | ~ Images (at ~2 ships/image) |
|---|---|---|
| ±5% | ~207 | ~100-110 |
| ±3% | ~575 | ~280-300 |
| ±2% | ~1290 | ~640-650 |

**Empirical validation.** The analytical estimate assumes independent samples
and ignores scene-level variance (background clutter, inshore/offshore mix),
so it was checked empirically: for each candidate subset size, draw 3
independent random samples (no replacement) from `HRSID_YOLO/images/val`, run
`model.val()` on each, and compare the mean and spread (std) of the metrics
against the full 1,962-image reference.

Script: [`sample_size_validation.py`](../sample_size_validation.py)

```bash
python sample_size_validation.py --device cpu
```

## Results

| Size | Precision (mean ± std) | Recall (mean ± std) | mAP@0.5 (mean ± std) |
|---|---|---|---|
| 100 | 0.889 ± 0.028 | 0.839 ± 0.020 | 0.854 ± 0.020 |
| 200 | 0.925 ± 0.017 | 0.834 ± 0.028 | 0.859 ± 0.022 |
| 300 | 0.911 ± 0.011 | 0.840 ± 0.044 | 0.858 ± 0.035 |
| 500 | 0.923 ± 0.007 | 0.857 ± 0.016 | 0.877 ± 0.008 |
| 800 | 0.917 ± 0.006 | 0.851 ± 0.014 | 0.872 ± 0.010 |
| 1200 | 0.910 ± 0.008 | 0.848 ± 0.011 | 0.865 ± 0.006 |
| **1962 (full set, reference)** | **0.913** | **0.846** | **0.863** |

Configuration: YOLO11m, `weights/best.pt`, imgsz=800, conf=0.25, iou=0.45,
device=cpu, 3 random trials per size, seed=0.

## Conclusion

Below 300 images, trial-to-trial variance is too large to trust a single run
(e.g. at 300 images, Recall swung between 0.797 and 0.885 across trials —
std=0.044). **From ~500 images onward, all three metrics stabilize within
±1.5 percentage points of the full-set reference, with std dropping to
0.6-1.6%.** Increasing further to 800-1200 images tightens the estimate only
marginally, showing diminishing returns past ~500 images.

**Recommendation:** a random sample of 500-800 images (25-40% of the full val
set) is sufficient to reliably estimate Precision, Recall, and mAP@0.5 for
this model/dataset, matching the analytical estimate (~280-650 for ±3-5%
margin of error). Samples should be drawn randomly (not just the first N
files) to preserve scene diversity (inshore/offshore mix, background clutter
variety).

## Notes

- The full-set reference here (Precision 0.913, Recall 0.846) matches
  [`results_inshore_offshore.md`](results_inshore_offshore.md) almost exactly,
  but mAP@0.5 differs (0.863 here vs. 0.918 there vs. 0.922 published) —
  consistent with the torch-version-dependent deviation already noted in that
  document (torch 2.12.0 here vs. 2.4.1 / 2.8.0 originally).
