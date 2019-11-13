// Low-level (BLAS-like) primitives.

#pragma once

#include "ctranslate2/devices.h"

namespace ctranslate2 {

  template <Device D = Device::CPU>
  struct primitives {

    static void set_device(int index);
    static int get_device();

    static void* alloc_data(size_t size);
    static void free_data(void* data);
    static void clear_cache();

    template <typename T>
    static T deref(const T* x, size_t index);

    template <typename T>
    static void fill(T* x, T a, size_t size);
    template <typename T>
    static void strided_fill(T* x, T a, size_t inc_x, size_t size);

    template <typename T>
    static void copy(const T* x, T* y, size_t size);

    template <typename T>
    static T sum(const T* array, size_t size);

    template <typename T>
    static T mean(const T* array, size_t size) {
      return sum(array, size) / size;
    }

    template <typename T>
    static size_t max_element(const T* array, size_t size);

    template <typename T>
    static T max(const T* array, size_t size);
    template <typename T>
    static T amax(const T* array, size_t size);

    template <typename T>
    static void add(T a, const T* x, T* y, size_t size);

    template <typename T>
    static void add(T a, T* y, size_t size) {
      add(a, y, y, size);
    }

    template <typename T>
    static void add(const T* a, const T* b, T* c, size_t size);

    template <typename T>
    static void add(const T* x, T* y, size_t size) {
      add(x, y, y, size);
    }

    template <typename T>
    static void add_batch_broadcast(const T* a, const T* b, T* c, size_t a_size, size_t b_size);

    template <typename T>
    static void add_batch_broadcast(const T* x, T* y, size_t x_size, size_t y_size) {
      add_batch_broadcast(x, y, y, x_size, y_size);
    }

    template <typename T>
    static void add_depth_broadcast(const T* a, const T* b, T* c, size_t a_size, size_t b_size);

    template <typename T>
    static void add_depth_broadcast(const T* x, T* y, size_t x_size, size_t y_size) {
      add_depth_broadcast(x, y, y, x_size, y_size);
    }

    template <typename T>
    static void sub(T a, const T* x, T* y, size_t size) {
      T a_rev = -a;
      add(a_rev, x, y, size);
    }

    template <typename T>
    static void sub(T a, T* y, size_t size) {
      sub(a, y, y, size);
    }

    template <typename T>
    static void sub(const T* a, const T* b, T* c, size_t size);

    template <typename T>
    static void mul(T a, const T* x, T* y, size_t size);

    template <typename T>
    static void mul(T a, T* y, size_t size) {
      mul(a, y, y, size);
    }

    template <typename T>
    static void mul_batch_broadcast(const T* a, const T* b, T* c, size_t a_size, size_t b_size);

    template <typename T>
    static void mul_batch_broadcast(const T* x, T* y, size_t x_size, size_t y_size) {
      mul_batch_broadcast(x, y, y, x_size, y_size);
    }

    template <typename T>
    static void mul_and_add_batch_broadcast(const T* x, const T* w, const T* b,
                                            T* y, size_t x_size, size_t wb_size);

    template <typename T>
    static void mul(const T* a, const T* b, T* c, size_t size);

    template <typename T>
    static void mul(const T* x, T* y, size_t size) {
      mul(x, y, y, size);
    }

    template <typename T>
    static void inv(const T* x, T* y, size_t size);

    template <typename T>
    static void quantize(const float* x, T* y, size_t size, float scale);

    template <typename T>
    static void unquantize(const T* x, float* y, size_t size, float scale);

    static void quantize_batch(const float* x, float* scales, int8_t* qx,
                               size_t batch_size, size_t depth);

    template <typename T>
    static void unquantize_batch(const T* x, const float* scale, float* y,
                                 size_t x_size, size_t scale_size);

    static void rescale_output(const int32_t* x,
                               const float* input_scales,
                               const float* weigh_scales,
                               float* y,
                               size_t batch_size,
                               size_t depth);

    template <typename DataType, typename IndexType>
    static void transpose_2d(const DataType* a, const IndexType* dims, DataType* b);

    template <typename DataType, typename IndexType>
    static void transpose_3d(const DataType* a,
                             const IndexType* dims,
                             const IndexType* perm,
                             DataType* b);

    template <typename DataType, typename IndexType>
    static void transpose_4d(const DataType* a,
                             const IndexType* dims,
                             const IndexType* perm,
                             DataType* b);

    static void pow(const float* x, float* y, float power, size_t size);
    static void exp(const float* x, float* y, size_t size);
    static void log(const float* x, float* y, size_t size);
    static void cos(const float* x, float* y, size_t size);
    static void sin(const float* x, float* y, size_t size);
    static void tanh(const float* x, float* y, size_t size);
    static void relu(const float* x, float* y, size_t size);
    static void gelu(const float* x, float* y, size_t size);

    template <typename In, typename Out>
    static void gemm(const In* a, const In* b,
                     bool transpose_a, bool transpose_b,
                     size_t m, size_t n, size_t k,
                     float alpha, float beta,
                     Out* c);

    template <typename In, typename Out>
    static void gemm_batch(const In* a, const In* b,
                           bool transpose_a, bool transpose_b,
                           size_t batch_size,
                           size_t m, size_t n, size_t k,
                           float alpha, float beta,
                           Out* c);
  };

  template <Device D1, Device D2>
  struct cross_device_primitives {
    template <typename T>
    static void copy(const T* x, T* y, size_t size);
  };

}
