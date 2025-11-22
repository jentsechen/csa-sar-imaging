cmake_minimum_required(VERSION 3.10)

project(MyCppProject VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(ZLIB_LIBRARY "C:/Users/user/Desktop/imaging_alg_dev/csa-sar-imaging/cnpy/zlib_deps/zlib.lib")
set(ZLIB_INCLUDE_DIR "C:/Users/user/Desktop/imaging_alg_dev/csa-sar-imaging/cnpy/zlib_deps")
set(CUSTOM_LIB_DIR "cnpy/build/Debug")
list(APPEND CMAKE_PREFIX_PATH ${CUSTOM_LIB_DIR})
find_package(ZLIB REQUIRED)
find_library(CNPY_LIBRARY NAMES cnpy REQUIRED HINTS ${CUSTOM_LIB_DIR}/lib)
# find_path(CNPY_INCLUDE_DIR cnpy.h REQUIRED HINTS ${CUSTOM_LIB_DIR}/include)

# set(FFTW3_LIBRARY "C:/Users/user/Desktop/imaging_alg_dev/csa-sar-imaging/fftw/libfftw3-3.lib")
# set(FFTW3_INCLUDE_DIR "C:/Users/user/Desktop/imaging_alg_dev/csa-sar-imaging/fftw")
# set(CUSTOM_LIB_DIR "fftw")
# list(APPEND CMAKE_PREFIX_PATH ${CUSTOM_LIB_DIR})
# find_package(FFTW3 REQUIRED)

set(FFTW_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fftw")
set(FFTW_INCLUDE_DIR "${FFTW_ROOT_DIR}")
set(FFTW_LIB_FILE "${FFTW_ROOT_DIR}/libfftw3-3.lib")
set(FFTW_DLL_FILE "${FFTW_ROOT_DIR}/libfftw3-3.dll")
add_library(FFTW3::fftw3 SHARED IMPORTED)
set_target_properties(FFTW3::fftw3 PROPERTIES
    IMPORTED_IMPLIB "${FFTW_LIB_FILE}"
)
set_target_properties(FFTW3::fftw3 PROPERTIES
    IMPORTED_LOCATION "${FFTW_DLL_FILE}"
)
set_target_properties(FFTW3::fftw3 PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIR}"
)

option(ENABLE_OPENMP "Enable OpenMP parallelization for this project" ON)
find_package(OpenMP REQUIRED)

add_executable(TestImagingPar src/TestImagingPar.cpp src/SigPar src/ImagingPar)
target_include_directories(TestImagingPar PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(TestImagingPar PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cnpy")
set_target_properties(TestImagingPar PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_link_libraries(TestImagingPar PRIVATE ${CNPY_LIBRARY} ZLIB::ZLIB FFTW3::fftw3 OpenMP::OpenMP_CXX)

add_executable(TestChirpScalingAlgo src/TestChirpScalingAlgo.cpp src/SigPar src/ImagingPar src/ChirpScalingAlgo)
target_include_directories(TestChirpScalingAlgo PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(TestChirpScalingAlgo PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cnpy")
target_include_directories(TestChirpScalingAlgo PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/fftw")
set_target_properties(TestChirpScalingAlgo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_link_libraries(TestChirpScalingAlgo PRIVATE ${CNPY_LIBRARY} ZLIB::ZLIB FFTW3::fftw3 OpenMP::OpenMP_CXX)
if(ENABLE_OPENMP)
    target_compile_definitions(TestChirpScalingAlgo PRIVATE USE_PARALLEL_FOR)
endif()

add_executable(TestIterRecovAlgo src/TestIterRecovAlgo.cpp src/SigPar src/ImagingPar src/ChirpScalingAlgo)
target_include_directories(TestIterRecovAlgo PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(TestIterRecovAlgo PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cnpy")
target_include_directories(TestIterRecovAlgo PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/fftw")
set_target_properties(TestIterRecovAlgo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_link_libraries(TestIterRecovAlgo PRIVATE ${CNPY_LIBRARY} ZLIB::ZLIB FFTW3::fftw3 OpenMP::OpenMP_CXX)