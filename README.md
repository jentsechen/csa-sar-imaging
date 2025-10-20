## How to Build (Windows)
```bash
mkdir build
cd build
rm ./CMakeCache.txt
cmake ..
cmake --build .
cmake --build . -j 4
```
* How to download CMake: https://hackmd.io/@YjqyYeVWSh27d4c92Ha5aw/r1LYBtiqL
* How to use json: https://github.com/nlohmann/json
    * Download json.hpp
    * Add include path to `c_cpp_properties.json`
    * Add `target_include_directories()` to `CMakeLists.txt`

## How to use CNPY
* Pre-request: ZLIB
```bash
cp -r ../zlib_deps/ .
```
* Git clone and CMake
```bash
git clone https://github.com/rogersce/cnpy.git
cd cnpy
cp ../zlib_deps
# revise CMakeLists.txt before install
mkdir build
cd build
rm CMakeCache.txt
cmake ..
cmake --install . --prefix ../install
```
* Revise CMakeLists.txt
    * Revise `CMAKE_MINIMUM_REQUIRED`: `CMAKE_MINIMUM_REQUIRED(VERSION 3.10 FATAL_ERROR)`
    * Add `ZLIB_LIBRARY` and `ZLIB_INCLUDE_DIR`
```bash
set(ZLIB_LIBRARY "C:/Users/user/Desktop/imaging_alg_dev/csa-sar-imaging/cnpy/zlib_deps/zlib.lib")
set(ZLIB_INCLUDE_DIR "C:/Users/user/Desktop/imaging_alg_dev/csa-sar-imaging/zlib_deps/cnpy")
```

## How to Run Unit Test of ImagingPar
```bash
cd TestImagingPar
```
* Test constructor
```bash
python "C:\Users\user\Desktop\imaging_alg_dev\csa-sar-imaging\TestImagingPar\TestConstructor.py"
```
* Test echo signal generation
```bash
python "C:\Users\user\Desktop\imaging_alg_dev\csa-sar-imaging\TestImagingPar\TestEchoSigGen.py"
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