#pragma once

#include <chrono>
#include <iostream>
#include <vector>

#ifdef CT2_WITH_CUDA
#  include <cuda_runtime.h>
#  define SYNCHRONIZE cudaDeviceSynchronize()
#else
#  define SYNCHRONIZE do {} while (false)
#endif

#define BENCHMARK(fun_call, samples)                                    \
  do {                                                                  \
    std::cerr << "benchmarking "#fun_call << std::endl;                 \
    for (size_t i = 0; i < 10; ++i)                                     \
      fun_call;                                                         \
    SYNCHRONIZE;                                                        \
    std::chrono::high_resolution_clock::time_point t1 =                 \
      std::chrono::high_resolution_clock::now();                        \
    for (size_t i = 0; i < static_cast<size_t>(samples); ++i)           \
      fun_call;                                                         \
    SYNCHRONIZE;                                                        \
    std::chrono::high_resolution_clock::time_point t2 =                 \
      std::chrono::high_resolution_clock::now();                        \
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count(); \
    std::cerr << "avg   "                                               \
      << static_cast<double>(duration) / (samples * 1000)               \
              << " ms" << std::endl;                                    \
  } while (false)


inline
std::vector<float> rand_vector(int size) {
  std::vector<float> vec(size);
  for (size_t i = 0; i < vec.size(); ++i)
    vec[i] = rand();
  return vec;
}

template <typename T>
double abs_diff(const std::vector<T>& a, const std::vector<T>& b) {
  double diff = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    diff += static_cast<double>(abs(a[i] - b[i]));
  }
  return diff;
}
