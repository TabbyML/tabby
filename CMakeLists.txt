cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 11)
set(MKL_DNN_ROOT /home/klein/dev/ctransformer/mkl-dnn/install)
set(MKL_ROOT /opt/intel/mkl)
set(TBB_ROOT /home/klein/downloads/tbb2018_20180312oss)
set(EIGEN_ROOT /home/klein/dev/ctransformer/eigen)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKL_ILP64 -m64 -mavx2 -Wall")
# find_package(Boost)
include_directories(
  # ${Boost_INCLUDE_DIRS}
  #${MKL_DNN_ROOT}/include
  ${MKL_ROOT}/include
  ${EIGEN_ROOT}
  #${TBB_ROOT}/include
  /home/klein/dev/ctransformer/gemmlowp/public
  )
link_directories(
  #${MKL_DNN_ROOT}/lib
  ${MKL_ROOT}/lib/intel64
  #${TBB_ROOT}/lib/intel64/gcc4.7/
  )
#add_definitions(-DEIGEN_USE_MKL_ALL)
set(MKL_LIBRARIES
  mkl_intel_ilp64
  mkl_gnu_thread
  mkl_core
  gomp
  #mkldnn
  pthread
  m
  dl
  )

add_executable(main
  compute_mkl.cc
  compute_simd.cc
  main.cc
  )
target_link_libraries(main
  ${MKL_LIBRARIES}
  )

# add_executable(benchmark
#   benchmark.cc
#   )
# target_link_libraries(benchmark
#   ${MKL_LIBRARIES}
#   )

add_executable(tensor_demo tensor_demo.cc)
target_link_libraries(tensor_demo
  ${MKL_LIBRARIES}
  )

add_subdirectory(tests)
