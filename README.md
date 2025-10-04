## How to Build (Windows)
```
mkdir build
cd build
cmake ..
cmake --build . -j 4
```
* How to download CMake: https://hackmd.io/@YjqyYeVWSh27d4c92Ha5aw/r1LYBtiqL
* How to use json: https://github.com/nlohmann/json
    * Download json.hpp
    * Add include path to `c_cpp_properties.json`
    * Add `target_include_directories()` to `CMakeLists.txt`

## How to Run Unit Test of ImagingPar
```
cd TestImagingPar
python "C:\Users\user\Desktop\imaging_alg_dev\csa-sar-imaging\TestImagingPar\TestImagingPar.py"
```
* Expected result
```
--- C++ Program Output (STDOUT) ---
Successfully parsed JSON from input_par.json
Successfully saved JSON to output_par.json

error of range_time_axis_sec: 1.3206826705950846e-16
error of range_freq_axis_hz: 0.0
error of azimuth_time_axis_sec: 0.0
error of azimuth_freq_axis_hz: 2.2724534198825808e-09
```