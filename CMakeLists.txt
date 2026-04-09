cmake_minimum_required(VERSION 3.10)

project(MyCppProject VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CNPY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cnpy")
find_library(CNPY_LIBRARY NAMES cnpy REQUIRED HINTS "${CNPY_DIR}/build")
find_path(CNPY_INCLUDE_DIR cnpy.h REQUIRED HINTS "${CNPY_DIR}")

option(ENABLE_OPENMP "Enable OpenMP parallelization for this project" ON)
find_package(OpenMP REQUIRED)

find_library(FFTW3_LIBRARY NAMES fftw3 REQUIRED
    HINTS /usr/lib/x86_64-linux-gnu /usr/local/lib "$ENV{HOME}/local/lib")
find_path(FFTW3_INCLUDE_DIR fftw3.h REQUIRED
    HINTS /usr/include /usr/local/include "${CMAKE_CURRENT_SOURCE_DIR}/fftw" "$ENV{HOME}/local/include")

set(COMMON_INCLUDES
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CNPY_INCLUDE_DIR}"
    "${FFTW3_INCLUDE_DIR}"
)

set(COMMON_LIBS
    ${FFTW3_LIBRARY}
    ${CNPY_LIBRARY}
    OpenMP::OpenMP_CXX
)

function(setup_sar_target TARGET_NAME)
    target_include_directories(${TARGET_NAME} PUBLIC ${COMMON_INCLUDES})
    target_link_libraries(${TARGET_NAME} PRIVATE ${COMMON_LIBS})
    
    if(ENABLE_OPENMP)
        target_compile_definitions(${TARGET_NAME} PRIVATE USE_PARALLEL_FOR)
    endif()
endfunction()

add_executable(TestImagingPar
    src/TestImagingPar.cpp
    src/SigPar.cpp
    src/ImagingPar.cpp
    src/PointTarget.cpp
    src/EchoSigGenPar.cpp
)
setup_sar_target(TestImagingPar)

add_executable(TestChirpScalingAlgo 
    src/TestChirpScalingAlgo.cpp 
    src/SigPar.cpp 
    src/ImagingPar.cpp 
    src/PointTarget.cpp 
    src/EchoSigGenPar.cpp
    src/ChirpScalingAlgo.cpp
)
setup_sar_target(TestChirpScalingAlgo)

add_executable(TestMultiPointTarget 
    src/TestMultiPointTarget.cpp 
    src/SigPar.cpp 
    src/ImagingPar.cpp 
    src/PointTarget.cpp 
    src/EchoSigGenPar.cpp
    src/ChirpScalingAlgo.cpp
)
setup_sar_target(TestMultiPointTarget)

add_executable(gen_echo_signal
    src/apps/gen_echo_signal.cpp
    src/SigPar.cpp
    src/ImagingPar.cpp
    src/PointTarget.cpp
    src/EchoSigGenPar.cpp
)
setup_sar_target(gen_echo_signal)

add_executable(linear_to_db_scale src/calc_acc/linear_to_db_scale.cpp)
setup_sar_target(linear_to_db_scale)