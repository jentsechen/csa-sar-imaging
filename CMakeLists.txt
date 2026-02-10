cmake_minimum_required(VERSION 3.10)

project(MyCppProject VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CUSTOM_LIB_DIR "cnpy/build/CMakeFiles")
list(APPEND CMAKE_PREFIX_PATH ${CUSTOM_LIB_DIR})
find_library(CNPY_LIBRARY NAMES cnpy REQUIRED HINTS ${CUSTOM_LIB_DIR}/lib)
find_path(CNPY_INCLUDE_DIR cnpy.h REQUIRED HINTS ${CUSTOM_LIB_DIR}/include)

option(ENABLE_OPENMP "Enable OpenMP parallelization for this project" ON)
find_package(OpenMP REQUIRED)

set(FFTW_LOCAL_DIR "$ENV{HOME}/jtc/local")
include_directories(${FFTW_LOCAL_DIR}/include)
link_directories(${FFTW_LOCAL_DIR}/lib)

set(COMMON_INCLUDES 
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/cnpy"
    "${FFTW_LOCAL_DIR}/include"
)

set(COMMON_LIBS 
    "${FFTW_LOCAL_DIR}/lib/libfftw3.so"
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
    src/TestImagingPar.cpp
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

add_executable(linear_to_db_scale src/calc_acc/linear_to_db_scale.cpp)
setup_sar_target(linear_to_db_scale)