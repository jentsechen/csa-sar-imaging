# sarsim — GPU SAR simulation + imaging pipeline

```
point_target_location/<stem>.json
  └─► echo.py      CUDA kernel (CuPy RawKernel)    echo stays on the GPU (never written)
        └─► imaging/<algo>.py   (CuPy / cuFFT)      same echo shared by every --algo
              └─► postproc.py   scene crop, |x|^2, 30 dB clip → JPG
                    └─► runs/<run>/<algo>/{jpg,power}/<stem>.*  + metrics.csv + manifest.json
```

| Module | Role |
|---|---|
| `params.py` | Axes / geometry (ports of `SigPar`, `ImagingPar`) and point-target list |
| `echo.py` | Thread-per-pixel echo kernel; ~0.14 s/scene on an RTX 5090 |
| `imaging/base.py` | `ImagingAlgorithm` interface + `@register` registry |
| `imaging/csa.py` | Chirp Scaling Algorithm; ~10–35 ms/scene |
| `postproc.py` | Crop, power, dB → uint8 (same as the original C++ path), metrics |
| `pipeline.py` | Batch driver |
| `validate.py` | Regression check against the C++ reference in `union_pipeline/reference/` |

## Usage

```bash
python -m sarsim.validate                                  # echo ~8e-9, CSA ~5e-16 vs C++
python -m sarsim.pipeline --run baseline_csa --algo csa    # all 1593 scenes, ~5 min
python -m sarsim.pipeline --run cmp --algo csa,rda --n 50  # several algorithms, same echoes
cd end_to_end_pipeline && python eval_union_csa.py --device 0 \
    --csa-dir union_pipeline/runs/baseline_csa/csa/jpg --name baseline_csa
```

The pipeline resumes (skips scenes whose JPGs exist for every requested
algorithm), refuses to append to a run made with a different `input_par`, and
pins itself away from CPU 1 (`--exclude-cpus`), a faulty core on this
workstation that causes random segfaults.

## Outputs (`end_to_end_pipeline/union_pipeline/runs/<run>/`)

- `<algo>/jpg/<stem>.jpg` — detector input (8-bit, 30 dB below the crop peak).
- `<algo>/power/<stem>.npy` — float32 **linear** `|x|^2` of the 800×800 scene
  crop, in the JPG's orientation (~2.5 MB). Use this, not the JPG, for speckle
  statistics (Phase 3) and despeckling (Phase 5): the JPG is clipped and
  quantized. dB is `10*log10(power)`.
- `metrics.csv` — per scene/algorithm: target count, echo / focus time, peak dB, entropy.
- `manifest.json` — `input_par`, git commit, algorithms, output formats.

Full-size echo / focused `.npy` files (164 MB each) are deliberately not
stored: regenerating an echo (0.14 s) is faster than reading one back from
`/home` (an SMR hard drive). Use `--save-echo <stem> ...` to dump specific
scenes for debugging.

## Adding an imaging algorithm

1. Create `imaging/<name>.py`, subclass `ImagingAlgorithm`, decorate with
   `@register("<name>")`, and add it to the imports in `imaging/__init__.py`.
2. Put every scene-independent computation (phase filters, interpolation
   kernels) in `__init__`; `focus(echo)` maps a complex GPU echo to a complex
   GPU image of the same shape.
3. FFT-based algorithms (RDA, ω-K) are straightforward CuPy. Per-pixel
   summations (back-projection) should be a RawKernel following `echo.py`'s
   thread-per-pixel pattern; vectorized CuPy would be ~70× slower there.
4. Validate on a single point target (resolution, PSLR, ISLR) before running
   the batch, then compare against CSA on the same run.
