cmake_minimum_required(VERSION 3.7)

# Set policy for setting the MSVC runtime library for static MSVC builds
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

project(ctranslate2)

option(WITH_MKL "Compile with Intel MKL backend" ON)
option(WITH_DNNL "Compile with DNNL backend" OFF)
option(WITH_ACCELERATE "Compile with Accelerate backend" OFF)
option(WITH_OPENBLAS "Compile with OpenBLAS backend" OFF)
option(WITH_RUY "Compile with Ruy backend" OFF)
option(WITH_CUDA "Compile with CUDA backend" OFF)
option(WITH_CUDNN "Compile with cuDNN backend" OFF)
option(CUDA_DYNAMIC_LOADING "Dynamically load CUDA libraries at runtime" OFF)
option(ENABLE_CPU_DISPATCH "Compile CPU kernels for multiple ISA and dispatch at runtime" ON)
option(ENABLE_PROFILING "Compile with profiling support" OFF)
option(BUILD_CLI "Compile the clients" ON)
option(BUILD_TESTS "Compile the tests" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

if(ENABLE_PROFILING)
  message(STATUS "Enable profiling support")
  add_definitions(-DCT2_ENABLE_PROFILING)
endif()

if(DEFINED ENV{INTELROOT})
  set(INTEL_ROOT_DEFAULT $ENV{INTELROOT})
elseif(DEFINED ENV{ONEAPI_ROOT})
  set(INTEL_ROOT_DEFAULT $ENV{ONEAPI_ROOT}/..)
elseif(DEFINED ENV{MKLROOT})
  set(INTEL_ROOT_DEFAULT $ENV{MKLROOT}/..)
elseif(WIN32)
  set(ProgramFilesx86 "ProgramFiles(x86)")
  set(INTEL_ROOT_DEFAULT PATHS
      $ENV{${ProgramFilesx86}}/IntelSWTools/compilers_and_libraries/windows
      $ENV{${ProgramFilesx86}}/Intel)
else()
  set(INTEL_ROOT_DEFAULT "/opt/intel")
endif()
set(INTEL_ROOT ${INTEL_ROOT_DEFAULT} CACHE FILEPATH "Path to Intel root directory")
set(OPENMP_RUNTIME "INTEL" CACHE STRING "OpenMP runtime (INTEL, COMP, NONE)")

# Set Release build type by default to get sane performance.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

# Set CXX flags.
set(CMAKE_CXX_STANDARD 17)

if(MSVC)
  if(BUILD_SHARED_LIBS)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  else()
    if(CMAKE_VERSION VERSION_LESS "3.15.0")
      message(FATAL_ERROR "Use CMake 3.15 or later when setting BUILD_SHARED_LIBS to OFF")
    endif()
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /d2FH4-")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

find_package(Threads)
add_subdirectory(third_party/spdlog EXCLUDE_FROM_ALL)

set(PUBLIC_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
set(PRIVATE_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  )
set(SOURCES
  src/allocator.cc
  src/batch_reader.cc
  src/buffered_translation_wrapper.cc
  src/cpu/allocator.cc
  src/cpu/backend.cc
  src/cpu/cpu_info.cc
  src/cpu/cpu_isa.cc
  src/cpu/kernels.cc
  src/cpu/primitives.cc
  src/decoding.cc
  src/decoding_utils.cc
  src/devices.cc
  src/env.cc
  src/generator.cc
  src/layers/attention.cc
  src/layers/common.cc
  src/layers/decoder.cc
  src/layers/transformer.cc
  src/layers/whisper.cc
  src/logging.cc
  src/models/language_model.cc
  src/models/model.cc
  src/models/model_reader.cc
  src/models/sequence_to_sequence.cc
  src/models/transformer.cc
  src/models/whisper.cc
  src/ops/activation.cc
  src/ops/add.cc
  src/ops/bias_add.cc
  src/ops/bias_add_cpu.cc
  src/ops/concat.cc
  src/ops/concat_split_cpu.cc
  src/ops/conv1d.cc
  src/ops/conv1d_cpu.cc
  src/ops/dequantize.cc
  src/ops/dequantize_cpu.cc
  src/ops/gather.cc
  src/ops/gather_cpu.cc
  src/ops/gelu.cc
  src/ops/gemm.cc
  src/ops/gumbel_max.cc
  src/ops/gumbel_max_cpu.cc
  src/ops/layer_norm.cc
  src/ops/layer_norm_cpu.cc
  src/ops/log.cc
  src/ops/matmul.cc
  src/ops/mean.cc
  src/ops/mean_cpu.cc
  src/ops/min_max.cc
  src/ops/mul.cc
  src/ops/multinomial.cc
  src/ops/multinomial_cpu.cc
  src/ops/quantize.cc
  src/ops/quantize_cpu.cc
  src/ops/relu.cc
  src/ops/rms_norm.cc
  src/ops/rms_norm_cpu.cc
  src/ops/softmax.cc
  src/ops/softmax_cpu.cc
  src/ops/split.cc
  src/ops/sub.cc
  src/ops/swish.cc
  src/ops/tanh.cc
  src/ops/tile.cc
  src/ops/tile_cpu.cc
  src/ops/topk.cc
  src/ops/topk_cpu.cc
  src/ops/transpose.cc
  src/padder.cc
  src/profiler.cc
  src/random.cc
  src/sampling.cc
  src/scoring.cc
  src/storage_view.cc
  src/thread_pool.cc
  src/translator.cc
  src/types.cc
  src/utils.cc
  src/vocabulary.cc
  src/vocabulary_map.cc
  )
set(LIBRARIES
  ${CMAKE_THREAD_LIBS_INIT}
  spdlog::spdlog_header_only
  )

macro(ct2_compile_kernels_for_isa isa flag)
  configure_file(
    src/cpu/kernels.cc
    ${CMAKE_CURRENT_BINARY_DIR}/kernels_${isa}.cc
    COPYONLY)
  set_source_files_properties(
    ${CMAKE_CURRENT_BINARY_DIR}/kernels_${isa}.cc
    PROPERTIES COMPILE_FLAGS ${flag})
  list(APPEND SOURCES ${CMAKE_CURRENT_BINARY_DIR}/kernels_${isa}.cc)
endmacro()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(arm64)|(aarch64)"
    OR (APPLE AND CMAKE_OSX_ARCHITECTURES STREQUAL "arm64"))
  add_definitions(-DCT2_ARM64_BUILD)
  set(CT2_BUILD_ARCH "arm64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(amd64)|(AMD64)")
  add_definitions(-DCT2_X86_BUILD)
  set(CT2_BUILD_ARCH "x86_64")

  if(BUILD_SHARED_LIBS)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  endif()
  set(BUILD_SHARED_LIBS_SAVED "${BUILD_SHARED_LIBS}")
  set(BUILD_SHARED_LIBS OFF)
  set(BUILD_TESTING OFF)
  add_subdirectory(third_party/cpu_features EXCLUDE_FROM_ALL)
  set(BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS_SAVED}")
  list(APPEND LIBRARIES cpu_features)
endif()

if(ENABLE_CPU_DISPATCH)
  message(STATUS "Compiling for multiple CPU ISA and enabling runtime dispatch")
  add_definitions(-DCT2_WITH_CPU_DISPATCH)
  if(CT2_BUILD_ARCH STREQUAL "x86_64")
    if(WIN32)
      ct2_compile_kernels_for_isa(avx "/arch:AVX")
      ct2_compile_kernels_for_isa(avx2 "/arch:AVX2")
      ct2_compile_kernels_for_isa(avx512 "/arch:AVX512")
    else()
      ct2_compile_kernels_for_isa(avx "-mavx")
      ct2_compile_kernels_for_isa(avx2 "-mavx2 -mfma")
      ct2_compile_kernels_for_isa(avx512 "-mavx512f -mavx512cd -mavx512vl -mavx512bw -mavx512dq")
    endif()
  elseif(CT2_BUILD_ARCH STREQUAL "arm64")
    ct2_compile_kernels_for_isa(neon "-DUSE_NEON")
  endif()
endif()

if(NOT OPENMP_RUNTIME STREQUAL "NONE")
  if(WIN32)
    add_compile_options("/openmp")
  else()
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
      add_compile_options(${OpenMP_CXX_FLAGS})
    endif()
  endif()

  if(OPENMP_RUNTIME STREQUAL "INTEL")
    # Find Intel libraries.
    find_library(IOMP5_LIBRARY iomp5 libiomp5md PATHS
      ${INTEL_ROOT}/lib
      ${INTEL_ROOT}/lib/intel64
      ${INTEL_ROOT}/compiler/lib/intel64
      ${INTEL_ROOT}/oneAPI/compiler/latest/windows/compiler/lib/intel64_win
      ${INTEL_ROOT}/oneapi/compiler/latest/linux/compiler/lib/intel64_lin
      ${INTEL_ROOT}/oneapi/compiler/latest/mac/compiler/lib
      )
    if(IOMP5_LIBRARY)
      list(APPEND LIBRARIES ${IOMP5_LIBRARY})
      message(STATUS "Using OpenMP: ${IOMP5_LIBRARY}")
    else()
      message(FATAL_ERROR "Intel OpenMP runtime libiomp5 not found")
    endif()
    if(WIN32)
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /nodefaultlib:vcomp")
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /nodefaultlib:vcomp")
    endif()
  elseif(OPENMP_RUNTIME STREQUAL "COMP")
    if(OpenMP_CXX_FOUND)
      list(APPEND LIBRARIES ${OpenMP_CXX_LIBRARIES})
      message(STATUS "Using OpenMP: ${OpenMP_CXX_LIBRARIES}")
    elseif(NOT WIN32)
      message(FATAL_ERROR "OpenMP not found")
    endif()
  else()
    message(FATAL_ERROR "Invalid OpenMP runtime ${OPENMP_RUNTIME}")
  endif()
endif()

if(WITH_MKL)
  find_path(MKL_ROOT include/mkl.h DOC "Path to MKL root directory" PATHS
    $ENV{MKLROOT}
    ${INTEL_ROOT}/mkl
    ${INTEL_ROOT}/oneAPI/mkl/latest
    ${INTEL_ROOT}/oneapi/mkl/latest
    )

  # Find MKL includes.
  find_path(MKL_INCLUDE_DIR NAMES mkl.h HINTS ${MKL_ROOT}/include/)
  if(MKL_INCLUDE_DIR)
    message(STATUS "Found MKL include directory: ${MKL_INCLUDE_DIR}")
  else()
    message(FATAL_ERROR "MKL include directory not found")
  endif()

  # Find MKL libraries.
  find_library(MKL_CORE_LIBRARY NAMES mkl_core PATHS ${MKL_ROOT}/lib ${MKL_ROOT}/lib/intel64)
  if(MKL_CORE_LIBRARY)
    get_filename_component(MKL_LIBRARY_DIR ${MKL_CORE_LIBRARY} DIRECTORY)
    message(STATUS "Found MKL library directory: ${MKL_LIBRARY_DIR}")
  else()
    message(FATAL_ERROR "MKL library directory not found")
  endif()

  add_definitions(-DCT2_WITH_MKL -DMKL_ILP64)
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
  elseif(OPENMP_RUNTIME STREQUAL "NONE")
    if(WIN32)
      list(APPEND MKL_LIBRARIES ${MKL_LIBRARY_DIR}/mkl_sequential.lib)
    else()
      list(APPEND MKL_LIBRARIES ${MKL_LIBRARY_DIR}/libmkl_sequential.a)
    endif()
  endif()
  list(APPEND PRIVATE_INCLUDE_DIRECTORIES ${MKL_INCLUDE_DIR})
  if(WIN32 OR APPLE)
    list(APPEND LIBRARIES ${MKL_LIBRARIES})
  else()
    list(APPEND LIBRARIES -Wl,--start-group ${MKL_LIBRARIES} -Wl,--end-group)
  endif()
endif()

if(WITH_DNNL)
  set(ONEAPI_DNNL_PATH ${INTEL_ROOT}/oneapi/dnnl/latest)
  if(OPENMP_RUNTIME STREQUAL "INTEL")
    set(ONEAPI_DNNL_PATH ${ONEAPI_DNNL_PATH}/cpu_iomp)
  else()
    set(ONEAPI_DNNL_PATH ${ONEAPI_DNNL_PATH}/cpu_gomp)
  endif()

  find_path(DNNL_INCLUDE_DIR NAMES dnnl.h PATHS ${ONEAPI_DNNL_PATH}/include)
  if(DNNL_INCLUDE_DIR)
    message(STATUS "Found DNNL include directory: ${DNNL_INCLUDE_DIR}")
  else()
    message(FATAL_ERROR "DNNL include directory not found")
  endif()

  find_library(DNNL_LIBRARY NAMES dnnl PATHS ${ONEAPI_DNNL_PATH}/lib)
  if(DNNL_LIBRARY)
    message(STATUS "Found DNNL library: ${DNNL_LIBRARY}")
  else()
    message(FATAL_ERROR "DNNL library not found")
  endif()

  add_definitions(-DCT2_WITH_DNNL)
  list(APPEND PRIVATE_INCLUDE_DIRECTORIES ${DNNL_INCLUDE_DIR})
  list(APPEND LIBRARIES ${DNNL_LIBRARY})
endif()

if (WITH_ACCELERATE)
  set(BLA_VENDOR Apple)
  find_package(BLAS REQUIRED)
  add_definitions(-DCT2_WITH_ACCELERATE)
  list(APPEND LIBRARIES ${BLAS_LIBRARIES})
endif()

if (WITH_OPENBLAS)
  find_path(OPENBLAS_INCLUDE_DIR NAMES cblas.h)
  if(OPENBLAS_INCLUDE_DIR)
    message(STATUS "Found OpenBLAS include directory: ${OPENBLAS_INCLUDE_DIR}")
  else()
    message(FATAL_ERROR "OpenBLAS include directory not found")
  endif()

  find_library(OPENBLAS_LIBRARY NAMES openblas)
  if(OPENBLAS_LIBRARY)
    message(STATUS "Found OpenBLAS library: ${OPENBLAS_LIBRARY}")
  else()
    message(FATAL_ERROR "OpenBLAS library not found")
  endif()

  add_definitions(-DCT2_WITH_OPENBLAS)
  list(APPEND PRIVATE_INCLUDE_DIRECTORIES ${OPENBLAS_INCLUDE_DIR})
  list(APPEND LIBRARIES ${OPENBLAS_LIBRARY})
endif()

if (WITH_RUY)
  add_definitions(-DCT2_WITH_RUY)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(CPUINFO_LIBRARY_TYPE static CACHE STRING "cpuinfo library type")
  add_subdirectory(third_party/ruy EXCLUDE_FROM_ALL)
  unset(CMAKE_POSITION_INDEPENDENT_CODE)
  list(APPEND LIBRARIES ruy)
endif()

if (WITH_CUDA)
  find_package(CUDA 11.0 REQUIRED)
  add_definitions(-DCT2_WITH_CUDA)
  if(MSVC)
    if(BUILD_SHARED_LIBS)
      list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=/MD$<$<CONFIG:Debug>:d>")
    else()
      list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=/MT$<$<CONFIG:Debug>:d>")
    endif()
  else()
    list(APPEND CUDA_NVCC_FLAGS "-std=c++17")
  endif()
  if(OpenMP_CXX_FOUND)
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=${OpenMP_CXX_FLAGS}")
  endif()

  if(NOT CUDA_ARCH_LIST)
    set(CUDA_ARCH_LIST "Auto")
  elseif(CUDA_ARCH_LIST STREQUAL "Common")
    set(CUDA_ARCH_LIST ${CUDA_COMMON_GPU_ARCHITECTURES})
    # Keep deprecated but not yet dropped Compute Capabilities.
    if(CUDA_VERSION_MAJOR EQUAL 11)
      list(INSERT CUDA_ARCH_LIST 0 "3.5" "5.0")
    endif()
    list(REMOVE_DUPLICATES CUDA_ARCH_LIST)
  endif()

  cuda_select_nvcc_arch_flags(ARCH_FLAGS ${CUDA_ARCH_LIST})
  list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
  set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

  message(STATUS "NVCC host compiler: ${CUDA_HOST_COMPILER}")
  message(STATUS "NVCC compilation flags: ${CUDA_NVCC_FLAGS}")

  # We should ensure that the Thrust include directories appear before
  # -I/usr/local/cuda/include for both GCC and NVCC, so that the headers
  # are coming from the submodule and not the system.
  set(THRUST_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/thrust/dependencies/cub
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/thrust
    )
  cuda_include_directories(${THRUST_INCLUDE_DIRS})
  list(APPEND PRIVATE_INCLUDE_DIRECTORIES ${THRUST_INCLUDE_DIRS})

  if(WITH_CUDNN)
    # Find cuDNN includes.
    find_path(CUDNN_INCLUDE_DIR NAMES cudnn.h HINTS ${CUDA_TOOLKIT_ROOT_DIR}/include)
    if(CUDNN_INCLUDE_DIR)
      message(STATUS "Found cuDNN include directory: ${CUDNN_INCLUDE_DIR}")
    else()
      message(FATAL_ERROR "cuDNN include directory not found")
    endif()

    # Find cuDNN libraries.
    find_library(CUDNN_LIBRARIES
      NAMES cudnn
      HINTS
      ${CUDA_TOOLKIT_ROOT_DIR}/lib
      ${CUDA_TOOLKIT_ROOT_DIR}/lib64
      ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
    )
    if(CUDNN_LIBRARIES)
      message(STATUS "Found cuDNN libraries: ${CUDNN_LIBRARIES}")
    else()
      message(FATAL_ERROR "cuDNN libraries not found")
    endif()

    # libcudnn.so is a shim layer that dynamically loads the correct library at runtime,
    # so we explictly link against it even with CUDA_DYNAMIC_LOADING.
    list(APPEND PRIVATE_INCLUDE_DIRECTORIES ${CUDNN_INCLUDE_DIR})
    list(APPEND LIBRARIES ${CUDNN_LIBRARIES})
    add_definitions(-DCT2_WITH_CUDNN)
  else()
    message(WARNING "cuDNN library is not enabled: convolution layers will not be supported on GPU")
  endif()

  if(CUDA_DYNAMIC_LOADING)
    list(APPEND SOURCES src/cuda/cublas_stub.cc)
  else()
    list(APPEND LIBRARIES ${CUDA_CUBLAS_LIBRARIES})
  endif()
  set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
  cuda_add_library(${PROJECT_NAME}
    ${SOURCES}
    src/cuda/allocator.cc
    src/cuda/primitives.cu
    src/cuda/random.cu
    src/cuda/utils.cc
    src/ops/bias_add_gpu.cu
    src/ops/concat_split_gpu.cu
    src/ops/conv1d_gpu.cu
    src/ops/dequantize_gpu.cu
    src/ops/gather_gpu.cu
    src/ops/gumbel_max_gpu.cu
    src/ops/layer_norm_gpu.cu
    src/ops/mean_gpu.cu
    src/ops/multinomial_gpu.cu
    src/ops/rms_norm_gpu.cu
    src/ops/softmax_gpu.cu
    src/ops/tile_gpu.cu
    src/ops/topk_gpu.cu
    src/ops/quantize_gpu.cu
    )
elseif(WITH_CUDNN)
  message(FATAL_ERROR "WITH_CUDNN=ON requires WITH_CUDA=ON")
else()
  add_library(${PROJECT_NAME} ${SOURCES})
endif()

list(APPEND LIBRARIES ${CMAKE_DL_LIBS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})
target_include_directories(${PROJECT_NAME} BEFORE
  PUBLIC ${PUBLIC_INCLUDE_DIRECTORIES}
  PRIVATE ${PRIVATE_INCLUDE_DIRECTORIES}
  )

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

include(GNUInstallDirs)

if (BUILD_CLI)
  add_subdirectory(cli)
endif()

install(
  TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h*"
  )
