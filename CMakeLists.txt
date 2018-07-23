cmake_minimum_required(VERSION 3.1.0)
project(ctranslate2)

option(LIB_ONLY "Do not compile clients" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

set(INTEL_ROOT /opt/intel)
set(MKL_ROOT ${INTEL_ROOT}/mkl)
set(MKL_SMALL_LIBRARY_NAME mkl_small)

# Set Release build type by default to get sane performance.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

# Set CXX flags.
set(CMAKE_CXX_STANDARD 11)
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Wall")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops -ffast-math")

find_package(Threads)

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

link_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${INTEL_LIBRARY_DIR}
  )

set(INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${MKL_INCLUDE_DIR}
  )
set(SOURCES
  src/decoder.cc
  src/decoder.cc
  src/model.cc
  src/primitives/cpu_mkl.cc
  src/storage_view.cc
  src/transformer.cc
  src/translation_result.cc
  src/translator.cc
  src/translator_pool.cc
  src/utils.cc
  src/vocabulary.cc
  src/vocabulary_map.cc
  )
set(LIBRARIES
  ${CMAKE_THREAD_LIBS_INIT}
  ${MKL_SMALL_LIBRARY_NAME}
  iomp5
  )

add_library(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIRECTORIES})

add_subdirectory(tests)

if (NOT LIB_ONLY)
  add_subdirectory(cli)
endif()

install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib/
  LIBRARY DESTINATION lib/
  )
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/lib${MKL_SMALL_LIBRARY_NAME}.so
  DESTINATION lib/
  )
install(
  DIRECTORY include/ctranslate2
  DESTINATION include
  FILES_MATCHING PATTERN "*.h*"
  )
