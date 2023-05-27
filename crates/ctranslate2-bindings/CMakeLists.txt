cmake_minimum_required(VERSION 3.7)

project(ctranslate2_bindings)

add_subdirectory(CTranslate2)

add_library(dummy
  src/dummy.cc
)

target_link_libraries(dummy
  PRIVATE ctranslate2
)

get_target_property(OUT ctranslate2 LINK_LIBRARIES)
message(STATUS ${OUT})

include(cmake/export_libs.cmake)
export_all_target_libs(dummy)
