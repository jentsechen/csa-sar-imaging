# HRSID Pixel Scale: Linear or Log, Amplitude or Intensity

Question: is an HRSID JPG pixel value (used directly as scatterer amplitude by the echo
simulation) (1) linear or logarithmic, and (2) amplitude or amplitude squared?

## Assumptions

- The images are the HRSID **JPG** release (`sar_ship_detect/HRSID_JPG/JPEGImages`), 8-bit,
  as used by the pipeline; the PNG release was not tested.
- Only offshore test scenes (`end_to_end_pipeline/images/`, 1593 scenes) are used.
- Sea = pixels farther than 10 px from every HRSID ship polygon; a 32x32 patch of sea
  is treated as homogeneous speckle (sea texture is ignored, which biases the look
  estimates).
- Speckle model: intensity ~ Gamma(L) (L-look, fully developed), amplitude = sqrt(intensity).
- Synthetic controls use JPEG quality 90; the quality of the HRSID JPGs is unknown.

## Product Specifications

- Sentinel-1 Level-1 GRD: the stored DN is amplitude; calibration squares it,
  β0 = DN² / A_β² (A_β from the product's calibration LUT), σ0 = β0 · sin(θ) [2, 3].
- TerraSAR-X / TanDEM-X detected products (MGD, GEC, EEC): β0 = k_s · DN², with the
  calibration factor k_s (`calFactor`) in the product metadata, so DN is amplitude [4].
- HRSID [1] is cropped from 136 Sentinel-1B, TerraSAR-X and TanDEM-X panoramas [5]. Neither the official README [6] nor any other source found states how the products were
  converted to 8-bit JPG; the PNG version is described only as "high-fidelity".

So the source data are amplitude, but the 8-bit conversion is undocumented and has to
be checked on the pixels.

## Statistical Test

`python3 scene_pixel_scale_test.py` (about 40 s): 975,864 homogeneous 32x32 sea
patches from all 1593 offshore scenes (fully farther than 10 px from every ship
polygon, no 255 pixels, no no-data areas). Every statistic is also computed on
synthetic speckle with the same patch mean levels, integer rounding and JPEG
compression (quality 90), generated as amplitude, as intensity and as dB of intensity.
Numbers: `pixel_scale.json`.

(1) Slope k of log(patch std) vs log(patch mean): speckle is multiplicative, so k ≈ 1
in linear units and k ≈ 0 after a log mapping.

(2) Two estimates of the number of looks L from the same patches, E[x²]/E[x]² and
Var(ln x); they agree only under the right hypothesis
(amplitude: E[A²]/E[A]² = LΓ(L)²/Γ(L+½)², Var(ln A) = ψ₁(L)/4;
intensity: E[I²]/E[I]² = 1 + 1/L, Var(ln I) = ψ₁(L)).

| data | slope k | L, amplitude hypothesis (ratio / log) | L, intensity hypothesis (ratio / log) |
|---|---|---|---|
| **HRSID sea** | **1.17** | **1.10 / 1.15** | 4.03 / 3.46 |
| synthetic amplitude, L = 1 | 1.12 | 0.91 / 0.97 | 3.32 / 2.78 |
| synthetic amplitude, L = 2 | 1.25 | 1.87 / 1.72 | 7.08 / 5.62 |
| synthetic amplitude, L = 4 | 1.32 | 4.61 / 4.36 | 17.98 / 16.03 |
| synthetic intensity, L = 1 | 1.03 | 0.30 / 0.59 | 1.00 / 1.51 |
| synthetic intensity, L = 2 | 1.04 | 0.53 / 0.76 | 1.87 / 2.05 |
| synthetic intensity, L = 4 | 1.13 | 0.98 / 1.08 | 3.60 / 3.19 |
| synthetic dB, L = 1 / 2 / 4 | 0.45 / 0.38 / 0.31 | | |

Per source panorama (123 panoramas with at least 200 patches): k median 1.18 (range
0.47-1.70); the amplitude hypothesis gives the more consistent pair of L estimates in
113 of 123 panoramas (median |ln(L₁/L₂)| 0.056 against 0.154 for intensity).

## Conclusions

1. **Linear, not log.** k = 1.17 overall and 1.18 per panorama, inside the range of the
   linear synthetics (1.03-1.32) and far from the dB synthetics (0.31-0.45, which stay
   above 0 only because of rounding and clipping near zero). Confirmed.
2. **Amplitude is favoured, not proven.** Under the amplitude hypothesis the two L
   estimates agree within 4 % (1.10 / 1.15), against 15 % under intensity, and the
   same holds in 113 of 123 panoramas; the real pattern matches synthetic
   single-look amplitude. Together with the product specifications (DN = amplitude),
   amplitude is the better-supported reading. The test is weak against 4-look
   intensity, which looks almost like 1-look amplitude (synthetic intensity L = 4
   gives 0.98 / 1.08 under the amplitude hypothesis), and sea texture biases both
   estimates, so this does not exclude a squared mapping on its own.
3. **Remaining caveat.** Under the amplitude reading the sea looks almost single-look
   (L ≈ 1.1). Multi-looked products (Sentinel-1 GRD) have more looks, so either texture
   lowers the apparent L or the data are close to single-look. Only the HRSID
   conversion step can settle this; the 10 panoramas that favour intensity and those
   with k far from 1 are where to look first.

## Sources

1. S. Wei et al., "HRSID: A High-Resolution SAR Images Dataset for Ship Detection and
   Instance Segmentation," IEEE Access, vol. 8, pp. 120234-120254, 2020.
   https://ieeexplore.ieee.org/document/9127939 (full text not accessed)
2. ESA, "Radiometric Calibration of Level-1 Products," Sentinel Online.
   https://sentinels.copernicus.eu/web/sentinel/radiometric-calibration-of-level-1-products
3. ESA STEP forum, β0 = DN²/A_β² for GRD (DN = amplitude).
   https://forum.step.esa.int/t/sentinel-1a-grd-product/291
4. Airbus Defence and Space, "Radiometric Calculations," TSXX-ITD-TN-0049 (β0 = k_s·DN²).
   https://rz-vm67.gfz-potsdam.de/attachments/download/365/r465_9_tsx-x-itd-tn-0049-radiometric_calculations_i3.00.pdf
   (link returned 404 when checked; formula taken from search excerpts)
5. Sensors of HRSID (Sentinel-1B, TerraSAR-X, TanDEM-X), as reported in
   https://arxiv.org/pdf/2103.08251
6. HRSID official repository. https://github.com/chaozhong2010/HRSID
