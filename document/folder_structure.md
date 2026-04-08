### Folder Structure
* Run: `git ls-tree -r --name-only HEAD | tree --fromfile .`
```text
.
├── .gitignore
├── CMakeLists.txt
├── README.md
├── TestChirpScalingAlgo
│   └── TestChirpScalingAlgo.py
├── TestImagingPar
│   ├── TestImagingPar.py
│   ├── azimuth_freq_axis_hz.csv
│   ├── azimuth_time_axis_sec.csv
│   ├── range_freq_axis_hz.csv
│   └── range_time_axis_sec.csv
├── TestMultiPointTarget
│   ├── AnalyzeIterResult.py
│   ├── TestImageGen.py
│   ├── TestMultiPointTarget.py
│   └── post_proc.py
├── diagram
│   ├── drawio
│   │   ├── image_to_echo.drawio
│   │   ├── image_to_echo.svg
│   │   ├── multi_point_target.drawio
│   │   └── multi_point_target.svg
│   ├── fixed_range_diff_azi_rng
│   │   ├── azi_slice.png
│   │   ├── image.png
│   │   └── rng_slice.png
│   ├── fixed_range_scene_center
│   │   ├── azi_slice.png
│   │   ├── image.png
│   │   └── rng_slice.png
│   ├── scaling
│   │   ├── coast_30_db_dynamic_range.png
│   │   ├── coast_full_scale.png
│   │   └── point_target_location.png
│   ├── speckle
│   │   ├── spatial_dom.png
│   │   ├── time_dom_azi_dim.png
│   │   └── time_dom_two_dim.png
│   ├── thresholding
│   │   └── island
│   │       ├── binary.png
│   │       ├── csa.png
│   │       ├── entropy.png
│   │       ├── n_non_zero_pixel.png
│   │       ├── original.png
│   │       ├── threshold_100.png
│   │       └── threshold_200.png
│   ├── var_range_diff_azi_rng
│   │   ├── azi_slice.png
│   │   ├── image.png
│   │   └── rng_slice.png
│   └── var_range_scene_center
│       ├── azi_slice.png
│       ├── image.png
│       └── rng_slice.png
├── document
│   ├── multi_point_target_sim_result.md
│   └── simulation_setup.md
├── fftw
│   ├── fftw3.h
│   ├── libfftw3-3.dll
│   └── libfftw3-3.lib
├── gen_raw_data
│   ├── gen_raw_data.py
│   ├── image_coast.png
│   └── image_island.png
├── include
│   ├── json.hpp
│   ├── util.cpp
│   └── util.h
└── src
    ├── ChirpScalingAlgo.cpp
    ├── ChirpScalingAlgo.h
    ├── EchoSigGenPar.cpp
    ├── EchoSigGenPar.h
    ├── ImagingPar.cpp
    ├── ImagingPar.h
    ├── PointTarget.cpp
    ├── PointTarget.h
    ├── SigPar.cpp
    ├── SigPar.h
    ├── TestChirpScalingAlgo.cpp
    ├── TestImagingPar.cpp
    ├── TestMultiPointTarget.cpp
    └── calc_acc
        └── linear_to_db_scale.cpp
```