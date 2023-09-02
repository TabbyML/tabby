cmake_minimum_required(VERSION 3.26)

set(CMAKE_CXX_STANDARD 14)

project(llama_cpp_bindings)

add_subdirectory(llama.cpp)

add_library(llama_cpp_bindings
  cc/lib.cc
)

target_link_libraries(llama_cpp_bindings
  PRIVATE llama
)
