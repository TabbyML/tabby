cmake_minimum_required(VERSION 3.1.0)
project(opennmt)

option(LIB_ONLY "Do not compile clients" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

set(CMAKE_CXX_STANDARD 11)

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Wall")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops -ffast-math")

if (NOT MKLML_ROOT)
  message(FATAL_ERROR "MKLML_ROOT must be defined")
endif()

add_definitions(-DWITH_MKL)
link_directories(${MKLML_ROOT}/lib)

find_package(Threads)

set(INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${MKLML_ROOT}/include
  )
set(SOURCES
  src/decoder.cc
  src/decoder.cc
  src/model.cc
  src/primitives/cpu_mkl.cc
  src/storage_view.cc
  src/transformer.cc
  src/translator.cc
  src/utils.cc
  src/vocabulary.cc
  src/vocabulary_map.cc
  )
set(LIBRARIES
  ${CMAKE_THREAD_LIBS_INIT}
  mklml_intel
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
  DIRECTORY include/opennmt
  DESTINATION include
  FILES_MATCHING PATTERN "*.h*"
  )
