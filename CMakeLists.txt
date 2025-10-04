cmake_minimum_required(VERSION 3.10)

project(MyCppProject VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(TestImagingPar src/TestImagingPar.cpp src/SigPar src/ImagingPar)
target_include_directories(TestImagingPar PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
set_target_properties(TestImagingPar PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")