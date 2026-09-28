# Progress Log

Weekly research log — append one entry per work session (2–3 lines max).
Format: date, what was done, any key decision or finding.

---

## 2026-05-31
- GitHub project structure created: 37 issues across 8 milestones, all labels set
- WBS finalized: 18–19 weeks without DL despeckle, ~21 with
- Next: start Phase 1 (speckle statistical analysis), run Phase 2 data collection in parallel

## 2026-09-28
- Echo generation: hand-written CUDA kernel (CuPy RawKernel) -> 1593 scenes in 4.3 min compute (was 4.3 h with vectorized CuPy); writing echo .npy is disk-bound (SMR HDD, ~261 GB)
- CuPy CSA port matches C++ to ~1e-15; fused GPU pipeline (echo -> CSA -> JPG) writes only JPGs
- Found all segfaults (TestMultiPointTarget + Python) occur on CPU 1 (core 4): faulty core, run jobs with `taskset -c 0,2-23`
- Restructured GPU code into `sarsim/` package (params / echo / imaging registry / postproc / pipeline / validate); run `baseline_csa` (1593 scenes, 4.7 min): P 0.9741 / R 0.9753 / mAP@0.5 0.9843 vs original 0.9818 / 0.9651 / 0.9838
- Deleted union_pipeline echo_signal/ + focused_image/ .npy (~670 GB); runs store JPG + float32 linear power crop only; C++ csa_jpg has one corrupt scene (P0062_3500_4300_1800_2600)
