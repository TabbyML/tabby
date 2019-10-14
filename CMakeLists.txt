cmake_minimum_required(VERSION 3.7)
project(ctranslate2)

option(WITH_MKL "Compile with Intel MKL backend" ON)
option(WITH_MKLDNN "Compile with Intel MKL-DNN (requires MKL)" OFF)
option(WITH_CUDA "Compile with CUDA backend" OFF)
option(LIB_ONLY "Do not compile clients" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

set(INTEL_ROOT /opt/intel CACHE FILEPATH "Path to Intel root directory")

# Set Release build type by default to get sane performance.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

# Set CXX flags.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -ffast-math")

find_package(Threads)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  add_compile_options(${OpenMP_CXX_FLAGS})
endif()

set(LINK_DIRECTORIES
  ${CMAKE_CURRENT_BINARY_DIR}
  )
set(INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
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
  src/ops/dequantize.cc
  src/ops/gather.cc
  src/ops/gemm.cc
  src/ops/layer_norm.cc
  src/ops/matmul.cc
  src/ops/mul.cc
  src/ops/relu.cc
  src/ops/softmax.cc
  src/ops/split.cc
  src/ops/sub.cc
  src/ops/tile.cc
  src/ops/topk.cc
  src/ops/transpose.cc
  src/ops/quantize.cc
  src/primitives/cpu.cc
  src/storage_view.cc
  src/translation_result.cc
  src/translator.cc
  src/translator_pool.cc
  src/utils.cc
  src/vocabulary.cc
  src/vocabulary_map.cc
  )
set(LIBRARIES
  ${CMAKE_DL_LIBS}
  ${CMAKE_THREAD_LIBS_INIT}
  )

if(WITH_MKL)
  set(MKL_ROOT ${INTEL_ROOT}/mkl)
  set(MKL_SMALL_LIBRARY_NAME mklml_intel)

  # Find MKL includes.
  find_path(MKL_INCLUDE_DIR mkl.h ${MKL_ROOT}/include/)
  if(MKL_INCLUDE_DIR)
    message(STATUS "Found MKL include directory: ${MKL_INCLUDE_DIR}")
  else()
    message(FATAL_ERROR "MKL include directory not found")
  endif()

  # Find Intel libraries.
  find_path(INTEL_LIBRARY_DIR libiomp5.so ${INTEL_ROOT}/lib/intel64)
  if(INTEL_LIBRARY_DIR)
    message(STATUS "Found Intel library directory: ${INTEL_LIBRARY_DIR}")
  else()
    message(FATAL_ERROR "Intel library directory not found")
  endif()

  # Generate custom MKL library.
  execute_process(
    COMMAND make
    libintel64
    name=${CMAKE_CURRENT_BINARY_DIR}/lib${MKL_SMALL_LIBRARY_NAME}
    export=${CMAKE_CURRENT_SOURCE_DIR}/mkl_symbol_list
    WORKING_DIRECTORY ${MKL_ROOT}/tools/builder
    OUTPUT_QUIET
    )

  if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/lib${MKL_SMALL_LIBRARY_NAME}.so)
    message(STATUS "Generated small MKL library lib${MKL_SMALL_LIBRARY_NAME}.so")
  else()
    message(FATAL_ERROR "An error occured when generating the MKL small library")
  endif()

  add_definitions(-DWITH_MKL)
  list(APPEND LINK_DIRECTORIES ${INTEL_LIBRARY_DIR})
  list(APPEND INCLUDE_DIRECTORIES ${MKL_INCLUDE_DIR})
  list(APPEND LIBRARIES
    ${MKL_SMALL_LIBRARY_NAME}
    iomp5
    )

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

link_directories(${LINK_DIRECTORIES})

if (WITH_CUDA)
  find_package(CUDA 10.0 REQUIRED)
  add_definitions(-DWITH_CUDA)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
  if(OpenMP_CXX_FOUND)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
  endif()

  if(NOT CUDA_ARCH_LIST)
    set(CUDA_ARCH_LIST "Auto")
  endif()

  cuda_select_nvcc_arch_flags(ARCH_FLAGS ${CUDA_ARCH_LIST})
  list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})

  message(STATUS "NVCC compilation flags: ${CUDA_NVCC_FLAGS}")

  find_path(CUDNN_LIBRARY_DIR libcudnn.so
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    /usr/lib/x86_64-linux-gnu
    )
  if(CUDNN_LIBRARY_DIR)
    message(STATUS "Found cuDNN library directory: ${CUDNN_LIBRARY_DIR}")
  else()
    message(FATAL_ERROR "cuDNN library directory not found")
  endif()
  link_directories(${CUDNN_LIBRARY_DIR})

  find_library(TENSORRT_LIBRARY NAMES nvinfer)
  find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h)
  if(NOT TENSORRT_LIBRARY OR NOT TENSORRT_INCLUDE_DIR)
    message(FATAL_ERROR "TensorRT not found")
  else()
    message(STATUS "Found TensorRT library: ${TENSORRT_LIBRARY}")
    message(STATUS "Found TensorRT include directory: ${TENSORRT_INCLUDE_DIR}")
  endif()

  find_path(CUB_INCLUDE_DIR NAMES cub/cub.cuh)
  if(NOT CUB_INCLUDE_DIR)
    message(FATAL_ERROR "CUB library not found")
  else()
    message(STATUS "Found CUB include directory: ${CUB_INCLUDE_DIR}")
  endif()

  list(APPEND INCLUDE_DIRECTORIES ${TENSORRT_INCLUDE_DIR})
  cuda_include_directories(
    ${INCLUDE_DIRECTORIES}
    ${CUB_INCLUDE_DIR}
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
    ${TENSORRT_LIBRARY}
    cudnn
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
  FILES "${CMAKE_CURRENT_BINARY_DIR}/lib${MKL_SMALL_LIBRARY_NAME}.so"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/ctranslate2"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h*"
  )
