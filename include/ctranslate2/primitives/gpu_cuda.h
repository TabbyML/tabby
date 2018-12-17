#pragma once

#include "primitives_decl.h"

namespace ctranslate2 {

  template<>
  void* primitives<Device::CUDA>::alloc_data(size_t size);
  template<>
  void primitives<Device::CUDA>::free_data(void* data);

  template<>
  template <typename T>
  T primitives<Device::CUDA>::deref(const T* x, size_t index) {
    T val = T();
    cross_device_primitives<Device::CUDA, Device::CPU>::copy(x + index, &val, 1);
    return val;
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::fill(T* x, T a, size_t size);
  template<>
  template <typename T>
  void primitives<Device::CUDA>::strided_fill(T* x, T a, size_t inc_x, size_t size);
  template<>
  template <typename T>
  void primitives<Device::CUDA>::copy(const T* x, T* y, size_t size);

  template<>
  template <typename T>
  T primitives<Device::CUDA>::sum(const T* array, size_t size);
  template<>
  template <typename T>
  size_t primitives<Device::CUDA>::max_element(const T* array, size_t size);
  template<>
  template <typename T>
  T primitives<Device::CUDA>::max(const T* array, size_t size);

  template<>
  template <typename T>
  void primitives<Device::CUDA>::add(T a, const T* x, T* y, size_t size);
  template<>
  template <typename T>
  void primitives<Device::CUDA>::add(const T* a, const T* b, T* c, size_t size);
  template<>
  template <typename T>
  void primitives<Device::CUDA>::add_batch_broadcast(const T* a, const T* b, T* c,
                                                     size_t a_size, size_t b_size);
  template<>
  template <typename T>
  void primitives<Device::CUDA>::add_depth_broadcast(const T* a, const T* b, T* c,
                                                     size_t a_size, size_t b_size);
  template<>
  template <typename T>
  void primitives<Device::CPU>::sub(const T* a, const T* b, T* c, size_t size);
  template<>
  template <typename T>
  void primitives<Device::CPU>::mul(T a, const T* x, T* y, size_t size);
  template<>
  template <typename T>
  void primitives<Device::CPU>::mul(const T* a, const T* b, T* c, size_t size);
  template<>
  template <typename T>
  void primitives<Device::CUDA>::mul_batch_broadcast(const T* a, const T* b, T* c,
                                                     size_t a_size, size_t b_size);

  template<>
  template<>
  void primitives<Device::CUDA>::relu(const float* x, float* y, size_t size);

  template<>
  template <typename DataType, typename IndexType>
  void primitives<Device::CUDA>::transpose_2d(const DataType* a, const IndexType* dims, DataType* b);
  template<>
  template <typename DataType, typename IndexType>
  void primitives<Device::CUDA>::transpose_3d(const DataType* a,
                                              const IndexType* dims,
                                              const IndexType* perm,
                                              DataType* b);
  template<>
  template <typename DataType, typename IndexType>
  void primitives<Device::CUDA>::transpose_4d(const DataType* a,
                                              const IndexType* dims,
                                              const IndexType* perm,
                                              DataType* b);

  template<>
  template<>
  void primitives<Device::CUDA>::gemm(const float* a, const float* b,
                                      bool transpose_a, bool transpose_b,
                                      size_t m, size_t n, size_t k,
                                      float alpha, float beta,
                                      float* c);

  template<>
  template<>
  void primitives<Device::CUDA>::gemm(const int8_t* a, const int8_t* b,
                                      bool transpose_a, bool transpose_b,
                                      size_t m, size_t n, size_t k,
                                      float alpha, float beta,
                                      int32_t* c);

  template<>
  template<>
  void primitives<Device::CUDA>::gemm_batch(const float* a, const float* b,
                                            bool transpose_a, bool transpose_b,
                                            size_t batch_size,
                                            size_t m, size_t n, size_t k,
                                            float alpha, float beta,
                                            float* c);

  template<>
  template<>
  void primitives<Device::CUDA>::exp(const float* x, float* y, size_t size);
  template<>
  template<>
  void primitives<Device::CUDA>::log(const float* x, float* y, size_t size);
  template<>
  template<>
  void primitives<Device::CUDA>::pow(const float* x, float* y, float power, size_t size);

  template<>
  template <typename T>
  void cross_device_primitives<Device::CPU, Device::CUDA>::copy(const T* x, T* y, size_t size);

  template<>
  template <typename T>
  void cross_device_primitives<Device::CUDA, Device::CPU>::copy(const T* x, T* y, size_t size);

}
