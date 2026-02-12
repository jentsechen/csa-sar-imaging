## Unsorted Notes
* How to use json: https://github.com/nlohmann/json
    * Download json.hpp
    * Add include path to `c_cpp_properties.json`
    * Add `target_include_directories()` to `CMakeLists.txt`

## How to Build (Linux)
```bash
git clone https://github.com/rogersce/cnpy.git
cd cnpy
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local
make
make install DESTDIR=$HOME/.local

cd ..
mkdir build # if build not exists
cd build
rm ./CMakeCache.txt # if CMakeCache.txt needs to be updated
cmake .. -DCNPY_LIBRARY=$HOME/.local/lib/libcnpy.so -DCNPY_INCLUDE_DIR=$HOME/.local/include
make
```

## How to Enable OpenMP
```bash
cd build
rm ./CMakeCache.txt
cmake .. -DENABLE_OPENMP=OFF
cmake --build .
```
* Performance of OpenMP acceleration
    * Time requirement of `ChirpScalingAlgo::ChirpScalingAlgo()` (constructor) from 42 seconds to 9 seconds
    * Time requirement of `ChirpScalingAlgo::apply_csa()` from 45 seconds to 30 seconds
    * Note that `fftw_execute()` can not be accelerated by directly using OpenMP 

## How to use CNPY
* Pre-request: ZLIB
    * https://zlib.net/
* Git clone and CMake
```bash
git clone https://github.com/rogersce/cnpy.git
cd cnpy
cp -r ../zlib_deps/ .
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

# Unit Tests
## How to Run Unit Test of ImagingPar
* How to run
```bash
cd TestImagingPar
python ./TestImagingPar.py
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
error of point_target_echo_signal: 9.769240521361663e-11
```
## How to Run Unit Test of ChirpScalingAlgo
* How to run
```bash
cd TestChirpScalingAlgo
python ./TestChirpScalingAlgo.py
```
* Expected result: mode = "par"
```
--- C++ Program Output (STDOUT) ---
Successfully parsed JSON from ../TestImagingPar/input_par.json
DONE

error of migr_par: 0.0
error of modified_range_fm_rate_hz_s: 0.85546875
error of chirp_scaling: 7.6016036847706285e-06
error of range_comp_filt: 5.650503436939742e-07
error of second_comp_filt: 1.51881785477161e-06
error of azimuth_comp_filt: 0.0005693735074238243
error of third_comp_filt: 2.8845511641389662e-05
```
* Expected result: mode = "app"
```
--- C++ Program Output (STDOUT) ---
Successfully parsed JSON from ../TestImagingPar/input_par.json
DONE

error of azimuth_fft_out: 2.534335101463643e-07
error of chirp_scaling_out: 3.878348030531107e-07
error of range_fft_out: 8.695627107279957e-05
error of second_phase_func_out: 0.0005971667732096324
error of range_ifft_out: 1.2526878838079562e-05
error of third_phase_func_out: 0.0007553788854503299
error of csa_out: 1.1978912833543335e-05
```
* Expected result: mode = "inverse_csa"
```
--- C++ Program Output (STDOUT) ---
Successfully parsed JSON from ../TestImagingPar/input_par.json
DONE

error of csa_out: 1.1978912833543335e-05
error of inverse_csa_out: 1.3500874256198407e-08
```

## File Not Upload to GitHub List
```
/csa-sar-imaging/TestImagingPar/echo_signal_golden.mat
/csa-sar-imaging/golden/
```

## Multi-Point Target
[Simulation Result](./document/multi_point_target_sim_result.md)

## Simulation Setup
[Simulation Setup](./document/simulation_setup.md)