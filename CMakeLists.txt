cmake_minimum_required(VERSION 3.1.0)
project(opennmt)

option(LIB_ONLY "Do not compile clients" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Wall")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

if (EIGEN_ROOT)
  set(ENV{EIGEN3_ROOT} ${EIGEN_ROOT})
endif()
if (EIGEN3_ROOT)
  set(ENV{EIGEN3_ROOT} ${EIGEN3_ROOT})
endif()

find_package(Eigen3 3.3 REQUIRED)

set(INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  )
set(SOURCES
  src/decoder.cc
  src/decoder.cc
  src/primitives/simd_primitives.cc
  src/storage_view.cc
  src/transformer.cc
  src/translator.cc
  src/vocabulary.cc
  )
set(LIBRARIES "")

find_package(MKL)

if (MKL_FOUND)
  add_definitions(-DINTEL_MKL)
  list(APPEND INCLUDE_DIRECTORIES ${MKL_INCLUDE_DIR})
  list(APPEND SOURCES src/primitives/mkl_primitives.cc)
  list(APPEND LIBRARIES ${MKL_LIBRARIES})
endif()

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
  DIRECTORY include/opennmt
  DESTINATION include
  FILES_MATCHING PATTERN "*.h*"
  )
