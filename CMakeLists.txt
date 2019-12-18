cmake_minimum_required(VERSION 3.7)

# Set policy for setting the MSVC runtime library for static MSVC builds
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

project(ctranslate2)

option(WITH_MKL "Compile with Intel MKL backend" ON)
option(WITH_MKLDNN "Compile with Intel MKL-DNN (requires MKL)" OFF)
option(WITH_CUDA "Compile with CUDA backend" OFF)
option(ENABLE_PROFILING "Compile with profiling support" OFF)
option(LIB_ONLY "Do not compile clients" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

if(ENABLE_PROFILING)
  message(STATUS "Enable profiling support")
  add_definitions(-DENABLE_PROFILING)
endif()

if(DEFINED ENV{INTELROOT})
  set(INTEL_ROOT_DEFAULT $ENV{INTELROOT})
elseif(DEFINED ENV{MKLROOT})
  set(INTEL_ROOT_DEFAULT $ENV{MKLROOT}/..)
elseif(WIN32)
  set(ProgramFilesx86 "ProgramFiles(x86)")
  set(INTEL_ROOT_DEFAULT $ENV{${ProgramFilesx86}}/IntelSWTools/compilers_and_libraries/windows)
else()
  set(INTEL_ROOT_DEFAULT "/opt/intel")
endif()
set(INTEL_ROOT ${INTEL_ROOT_DEFAULT} CACHE FILEPATH "Path to Intel root directory")
set(OPENMP_RUNTIME "INTEL" CACHE STRING "OpenMP runtime (INTEL, COMP)")

# Set Release build type by default to get sane performance.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

# Set CXX flags.
set(CMAKE_CXX_STANDARD 11)

if(MSVC)
  if(BUILD_SHARED_LIBS)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  else()
    if(CMAKE_VERSION VERSION_LESS "3.15.0")
      message(FATAL_ERROR "Use CMake 3.15 or later when setting BUILD_SHARED_LIBS to OFF")
    endif()
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -ffast-math")
endif()

find_package(Threads)

set(INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
set(SOURCES
  src/decoding.cc
  src/devices.cc
  src/layers/attention.cc
  src/layers/decoder.cc
  src/layers/common.cc
  src/models/model.cc
  src/models/transformer.cc
  src/ops/add.cc
  src/ops/concat.cc
  src/ops/concat_cpu.cc
  src/ops/dequantize.cc
  src/ops/gather.cc
  src/ops/gather_cpu.cc
  src/ops/gelu.cc
  src/ops/gemm.cc
  src/ops/layer_norm.cc
  src/ops/layer_norm_cpu.cc
  src/ops/matmul.cc
  src/ops/mul.cc
  src/ops/relu.cc
  src/ops/softmax.cc
  src/ops/softmax_cpu.cc
  src/ops/split.cc
  src/ops/split_cpu.cc
  src/ops/sub.cc
  src/ops/tile.cc
  src/ops/topk.cc
  src/ops/topk_cpu.cc
  src/ops/transpose.cc
  src/ops/quantize.cc
  src/primitives/cpu.cc
  src/profiler.cc
  src/storage_view.cc
  src/translation_result.cc
  src/translator.cc
  src/translator_pool.cc
  src/types.cc
  src/utils.cc
  src/vocabulary.cc
  src/vocabulary_map.cc
  )
set(LIBRARIES
  ${CMAKE_DL_LIBS}
  ${CMAKE_THREAD_LIBS_INIT}
  )

if(NOT WIN32)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    add_compile_options(${OpenMP_CXX_FLAGS})
  endif()
endif()

if(OPENMP_RUNTIME STREQUAL "INTEL")
  # Find Intel libraries.
  find_library(IOMP5_LIBRARY iomp5 libiomp5md HINTS ${INTEL_ROOT}/lib/intel64 ${INTEL_ROOT}/compiler/lib/intel64)
  if(IOMP5_LIBRARY)
    list(APPEND LIBRARIES ${IOMP5_LIBRARY})
    message(STATUS "Using OpenMP: ${IOMP5_LIBRARY}")
  else()
    message(FATAL_ERROR "Intel OpenMP runtime libiomp5 not found")
  endif()
elseif(OPENMP_RUNTIME STREQUAL "COMP")
  if(OpenMP_CXX_FOUND)
    list(APPEND LIBRARIES ${OpenMP_CXX_LIBRARIES})
    message(STATUS "Using OpenMP: ${OpenMP_CXX_LIBRARIES}")
  else()
    message(FATAL_ERROR "OpenMP not found")
  endif()
else()
  message(FATAL_ERROR "Invalid OpenMP runtime ${OPENMP_RUNTIME}")
endif()

if(WITH_MKL)
  find_path(MKL_ROOT include/mkl.h PATHS $ENV{MKLROOT} ${INTEL_ROOT}/mkl
                                   DOC "Path to MKL root directory")

  # Find MKL includes.
  find_path(MKL_INCLUDE_DIR NAMES mkl.h HINTS ${MKL_ROOT}/include/)
  if(MKL_INCLUDE_DIR)
    message(STATUS "Found MKL include directory: ${MKL_INCLUDE_DIR}")
  else()
    message(FATAL_ERROR "MKL include directory not found")
  endif()

  # Find MKL libraries.
  find_library(MKL_CORE_LIBRARY NAMES mkl_core PATHS ${MKL_ROOT}/lib/intel64)
  if(MKL_CORE_LIBRARY)
    get_filename_component(MKL_LIBRARY_DIR ${MKL_CORE_LIBRARY} DIRECTORY)
    message(STATUS "Found MKL library directory: ${MKL_LIBRARY_DIR}")
  else()
    message(FATAL_ERROR "MKL library directory not found")
  endif()

  add_definitions(-DWITH_MKL -DMKL_ILP64)
  if(WIN32)
    set(MKL_LIBRARIES
      ${MKL_LIBRARY_DIR}/mkl_core.lib
      ${MKL_LIBRARY_DIR}/mkl_intel_ilp64.lib
      )
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
    set(MKL_LIBRARIES
      ${MKL_LIBRARY_DIR}/libmkl_core.a
      ${MKL_LIBRARY_DIR}/libmkl_intel_ilp64.a
      )
  endif()

  if(OPENMP_RUNTIME STREQUAL "INTEL")
    if(WIN32)
      list(APPEND MKL_LIBRARIES ${MKL_LIBRARY_DIR}/mkl_intel_thread.lib)
    else()
      list(APPEND MKL_LIBRARIES ${MKL_LIBRARY_DIR}/libmkl_intel_thread.a)
    endif()
  elseif(OPENMP_RUNTIME STREQUAL "COMP")
    if(WIN32)
      message(FATAL_ERROR "Building with MKL requires Intel OpenMP")
    else()
      list(APPEND MKL_LIBRARIES ${MKL_LIBRARY_DIR}/libmkl_gnu_thread.a)
    endif()
  endif()
  list(APPEND INCLUDE_DIRECTORIES ${MKL_INCLUDE_DIR})
  if(WIN32)
    list(APPEND LIBRARIES ${MKL_LIBRARIES})
  else()
    list(APPEND LIBRARIES -Wl,--start-group ${MKL_LIBRARIES} -Wl,--end-group)
  endif()

  if(WITH_MKLDNN)
    find_path(MKLDNN_INCLUDE_DIR mkldnn.hpp)
    if(MKLDNN_INCLUDE_DIR)
      message(STATUS "Found MKL-DNN include directory: ${MKLDNN_INCLUDE_DIR}")
    else()
      message(FATAL_ERROR "MKL-DNN include directory not found")
    endif()

    find_library(MKLDNN_LIBRARY mkldnn)
    if(MKLDNN_LIBRARY)
      message(STATUS "Found MKL-DNN library: ${MKLDNN_LIBRARY}")
    else()
      message(FATAL_ERROR "MKL-DNN library not found")
    endif()

    add_definitions(-DWITH_MKLDNN)
    list(APPEND INCLUDE_DIRECTORIES ${MKLDNN_INCLUDE_DIR})
    list(APPEND LIBRARIES ${MKLDNN_LIBRARY})
  endif()
elseif(WITH_MKLDNN)
  message(FATAL_ERROR "Building with MKL-DNN requires MKL")
endif()

if (WITH_CUDA)
  find_package(CUDA 10.0 REQUIRED)
  add_definitions(-DWITH_CUDA)
  if(MSVC)
    if(BUILD_SHARED_LIBS)
      set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler=/MD$<$<CONFIG:Debug>:d>")
    else()
      set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler=/MT$<$<CONFIG:Debug>:d>")
    endif()
  else()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
  endif()
  if(OpenMP_CXX_FOUND)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
  endif()

  if(NOT CUDA_ARCH_LIST)
    set(CUDA_ARCH_LIST "Auto")
  endif()

  cuda_select_nvcc_arch_flags(ARCH_FLAGS ${CUDA_ARCH_LIST})
  list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})

  find_library(CUDNN_LIBRARY cudnn
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
    /usr/lib/x86_64-linux-gnu
    )
  if(CUDNN_LIBRARY)
    message(STATUS "Found cuDNN library: ${CUDNN_LIBRARY}")
  else()
    message(FATAL_ERROR "cuDNN library not found")
  endif()

  find_library(TENSORRT_LIBRARY NAMES nvinfer)
  find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h)
  if(NOT TENSORRT_LIBRARY OR NOT TENSORRT_INCLUDE_DIR)
    message(FATAL_ERROR "TensorRT not found")
  else()
    message(STATUS "Found TensorRT library: ${TENSORRT_LIBRARY}")
    message(STATUS "Found TensorRT include directory: ${TENSORRT_INCLUDE_DIR}")
  endif()

  if(NOT MSVC)
    # TensorRT 6 header generates a lot of deprecating warnings. Ignore them.
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -isystem ${TENSORRT_INCLUDE_DIR}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${TENSORRT_INCLUDE_DIR}")
  endif()

  message(STATUS "NVCC compilation flags: ${CUDA_NVCC_FLAGS}")

  find_path(CUB_INCLUDE_DIR NAMES cub/cub.cuh)
  if(NOT CUB_INCLUDE_DIR)
    message(FATAL_ERROR "CUB library not found")
  else()
    message(STATUS "Found CUB include directory: ${CUB_INCLUDE_DIR}")
  endif()

  find_path(THRUST_INCLUDE_DIR
    NAMES thrust/device_vector.h
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/include)
  if(NOT THRUST_INCLUDE_DIR)
    message(FATAL_ERROR "Thrust library not found")
  else()
    message(STATUS "Found Thrust include directory: ${THRUST_INCLUDE_DIR}")
  endif()

  include_directories(
    ${TENSORRT_INCLUDE_DIR}
  )

  cuda_include_directories(
    ${CUB_INCLUDE_DIR}
    ${THRUST_INCLUDE_DIR}
    )
  cuda_add_library(${PROJECT_NAME}
    ${SOURCES}
    src/cuda/utils.cc
    src/primitives/cuda.cu
    src/ops/concat_split_gpu.cu
    src/ops/gather_gpu.cu
    src/ops/layer_norm_gpu.cu
    src/ops/softmax_gpu.cu
    src/ops/topk_gpu.cu
    )
  list(APPEND LIBRARIES
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDNN_LIBRARY}
    ${TENSORRT_LIBRARY}
    )
else()
  add_library(${PROJECT_NAME} ${SOURCES})
endif()

target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIRECTORIES})

add_subdirectory(tests)

include(GNUInstallDirs)

if (NOT LIB_ONLY)
  add_subdirectory(cli)
endif()

install(
  TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/ctranslate2"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h*"
  )
