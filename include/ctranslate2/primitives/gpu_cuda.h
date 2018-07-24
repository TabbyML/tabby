#pragma once

#include "primitives.h"

namespace ctranslate2 {

  template<>
  void* primitives<Device::CUDA>::alloc_data(size_t size);
  template<>
  void primitives<Device::CUDA>::free_data(void* data);

  template<>
  template <typename T>
  void primitives<Device::CUDA>::fill(T* x, T a, size_t size);
  template<>
  template <typename T>
  void primitives<Device::CUDA>::copy(const T* x, T* y, size_t size);

  template<>
  template<>
  void primitives<Device::CUDA>::gemm(const float* a, const float* b,
                                      bool transpose_a, bool transpose_b,
                                      size_t m, size_t n, size_t k,
                                      float alpha, float beta,
                                      float* c);

  template<>
  template<>
  void primitives<Device::CUDA>::gemm_batch(const float* a, const float* b,
                                            bool transpose_a, bool transpose_b,
                                            size_t batch_size,
                                            size_t m, size_t n, size_t k,
                                            float alpha, float beta,
                                            float* c);

  template<>
  template <typename T>
  void cross_device_primitives<Device::CPU, Device::CUDA>::copy(const T* x, T* y, size_t size);

  template<>
  template <typename T>
  void cross_device_primitives<Device::CUDA, Device::CPU>::copy(const T* x, T* y, size_t size);

}
