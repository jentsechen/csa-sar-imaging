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

add_executable(TestImagingPar src/TestImagingPar.cpp src/SigPar src/ImagingPar)
target_include_directories(TestImagingPar PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(TestImagingPar PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cnpy")
set_target_properties(TestImagingPar PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_link_libraries(TestImagingPar PRIVATE ${CNPY_LIBRARY} ZLIB::ZLIB)

add_executable(TestChirpScalingAlgo src/TestChirpScalingAlgo.cpp src/SigPar src/ImagingPar src/ChirpScalingAlgo)
target_include_directories(TestChirpScalingAlgo PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(TestChirpScalingAlgo PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cnpy")
set_target_properties(TestChirpScalingAlgo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_link_libraries(TestChirpScalingAlgo PRIVATE ${CNPY_LIBRARY} ZLIB::ZLIB)