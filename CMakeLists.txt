cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 11)
set(MKL_DNN_ROOT /home/klein/dev/ctransformer/mkl-dnn/install)
set(MKL_ROOT /opt/intel/mkl)
set(EIGEN_ROOT /home/klein/dev/ctransformer/eigen)
set(CMAKE_BUILD_TYPE Release)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
# find_package(Boost)
include_directories(
  # ${Boost_INCLUDE_DIRS}
  ${MKL_DNN_ROOT}/include
  ${MKL_ROOT}/include
  ${EIGEN_ROOT}
  /home/klein/dev/ctransformer/gemmlowp/public
  )
link_directories(
  ${MKL_DNN_ROOT}/lib
  ${MKL_ROOT}/lib/intel64
  )
#add_definitions(-DEIGEN_USE_MKL_ALL)
add_executable(main
  main.cc
  )
add_executable(benchmark
  benchmark.cc
  )
target_link_libraries(main
  mkl_rt
  mkldnn
  )
target_link_libraries(benchmark
  mkl_rt
  mkldnn
  )
