cmake_minimum_required(VERSION 3.7)

project(ctranslate2_bindings)

add_subdirectory(CTranslate2)

add_library(dummy
  src/dummy.cc
)

target_link_libraries(dummy
  PRIVATE ctranslate2
)

include(../rust-cxx-cmake-bridge/cmake/export_libs.cmake)
export_all_target_libs(dummy)
