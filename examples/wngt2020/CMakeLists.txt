cmake_minimum_required(VERSION 3.7)
project(wngt2020)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
find_package(Threads)
add_executable(run main.cc)
target_link_libraries(run PRIVATE
  ${CMAKE_THREAD_LIBS_INIT}
  ctranslate2
  sentencepiece
  ${CMAKE_DL_LIBS}
  )
