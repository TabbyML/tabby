#pragma once

#include "cpu_generic.h"

namespace ctranslate2 {

  // MKL specializations.

  template<>
  void* primitives<Device::CPU>::alloc_data(size_t size);
  template<>
  void primitives<Device::CPU>::free_data(void* data);

  template<>
  template<>
  void primitives<Device::CPU>::copy(const float* x, float* y, size_t size);

  template<>
  template<>
  void primitives<Device::CPU>::add(const float* a, const float* b, float* c, size_t size);
  template<>
  template<>
  void primitives<Device::CPU>::sub(const float* a, const float* b, float* c, size_t size);
  template<>
  template<>
  void primitives<Device::CPU>::mul(float a, float* y, size_t size);
  template<>
  template<>
  void primitives<Device::CPU>::mul(const float* a, const float* b, float* c, size_t size);

  template<>
  template<>
  void primitives<Device::CPU>::inv(const float* x, float* y, size_t size);
  template<>
  template<>
  void primitives<Device::CPU>::pow(const float* x, float *y, float power, size_t size);
  template<>
  template<>
  void primitives<Device::CPU>::exp(const float* x, float* y, size_t size);
  template<>
  template<>
  void primitives<Device::CPU>::log(const float* x, float* y, size_t size);
  template<>
  template<>
  void primitives<Device::CPU>::tanh(const float* x, float* y, size_t size);

  template<>
  template<>
  void primitives<Device::CPU>::transpose_2d(const float* a, const size_t* dims, float* b);

  template<>
  template<>
  void primitives<Device::CPU>::gemm(const float* a, const float* b,
                                     bool transpose_a, bool transpose_b,
                                     size_t m, size_t n, size_t k,
                                     float alpha, float beta,
                                     float* c);
  template<>
  template<>
  void primitives<Device::CPU>::gemm(const int16_t* a, const int16_t* b,
                                     bool transpose_a, bool transpose_b,
                                     size_t m, size_t n, size_t k,
                                     float alpha, float beta,
                                     int32_t* c);

  template<>
  template<>
  void primitives<Device::CPU>::gemm(const int8_t* a, const int8_t* b,
                                     bool transpose_a, bool transpose_b,
                                     size_t m, size_t n, size_t k,
                                     float alpha, float beta,
                                     int32_t* c);

  template<>
  template<>
  void primitives<Device::CPU>::gemm_batch(const float* a, const float* b,
                                           bool transpose_a, bool transpose_b,
                                           size_t batch_size,
                                           size_t m, size_t n, size_t k,
                                           float alpha, float beta,
                                           float* c);

}
