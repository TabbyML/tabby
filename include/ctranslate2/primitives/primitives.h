// Low-level (BLAS-like) primitives.

#pragma once

#include "ctranslate2/devices.h"
#include "ctranslate2/types.h"

namespace ctranslate2 {

  template <Device D = Device::CPU>
  struct primitives {

    static void set_device(int index);
    static int get_device();

    static void* alloc_data(dim_t size, int device_index = -1);
    static void free_data(void* data, int device_index = -1);
    static void clear_cache();

    template <typename T>
    static T deref(const T* x, dim_t index);

    template <typename T>
    static void fill(T* x, T a, dim_t size);
    template <typename T>
    static void strided_fill(T* x, T a, dim_t inc_x, dim_t size);

    template <typename T>
    static void copy(const T* x, T* y, dim_t size);

    template <typename T>
    static T sum(const T* array, dim_t size);

    template <typename T>
    static T mean(const T* array, dim_t size) {
      return sum(array, size) / size;
    }

    template <typename T>
    static dim_t max_element(const T* array, dim_t size);

    template <typename T>
    static T max(const T* array, dim_t size);
    template <typename T>
    static T amax(const T* array, dim_t size);
    template <typename T>
    static void row_max(const T* x,
                        const dim_t rows,
                        const dim_t cols,
                        T* values,
                        int32_t* indices);

    template <typename T>
    static void add(T a, const T* x, T* y, dim_t size);

    template <typename T>
    static void add(T a, T* y, dim_t size) {
      add(a, y, y, size);
    }

    template <typename T>
    static void add(const T* a, const T* b, T* c, dim_t size);

    template <typename T>
    static void add(const T* x, T* y, dim_t size) {
      add(x, y, y, size);
    }

    template <typename T>
    static void add_batch_broadcast(const T* a, const T* b, T* c, dim_t a_size, dim_t b_size);

    template <typename T>
    static void add_batch_broadcast(const T* x, T* y, dim_t x_size, dim_t y_size) {
      add_batch_broadcast(x, y, y, x_size, y_size);
    }

    template <typename T>
    static void add_depth_broadcast(const T* a, const T* b, T* c, dim_t a_size, dim_t b_size);

    template <typename T>
    static void add_depth_broadcast(const T* x, T* y, dim_t x_size, dim_t y_size) {
      add_depth_broadcast(x, y, y, x_size, y_size);
    }

    template <typename T>
    static void sub(T a, const T* x, T* y, dim_t size) {
      T a_rev = -a;
      add(a_rev, x, y, size);
    }

    template <typename T>
    static void sub(T a, T* y, dim_t size) {
      sub(a, y, y, size);
    }

    template <typename T>
    static void sub(const T* a, const T* b, T* c, dim_t size);

    template <typename T>
    static void max(T a, const T* x, T* y, dim_t size);

    template <typename T>
    static void max(const T* a, const T* b, T* c, dim_t size);

    template <typename T>
    static void max(T a, T* y, dim_t size) {
      max(a, y, y, size);
    }

    template <typename T>
    static void min(T a, const T* x, T* y, dim_t size);

    template <typename T>
    static void min(const T* a, const T* b, T* c, dim_t size);

    template <typename T>
    static void min(T a, T* y, dim_t size) {
      min(a, y, y, size);
    }

    template <typename T>
    static void mul(T a, const T* x, T* y, dim_t size);

    template <typename T>
    static void mul(T a, T* y, dim_t size) {
      mul(a, y, y, size);
    }

    template <typename T>
    static void mul_batch_broadcast(const T* a, const T* b, T* c, dim_t a_size, dim_t b_size);

    template <typename T>
    static void mul_batch_broadcast(const T* x, T* y, dim_t x_size, dim_t y_size) {
      mul_batch_broadcast(x, y, y, x_size, y_size);
    }

    template <typename T>
    static void mul(const T* a, const T* b, T* c, dim_t size);

    template <typename T>
    static void mul(const T* x, T* y, dim_t size) {
      mul(x, y, y, size);
    }

    template <typename T>
    static void quantize(const float* x, T* y, dim_t size, float scale, float shift = 0);

    template <typename T>
    static void dequantize(const T* x, float* y, dim_t size, float scale, float shift = 0);

    static void quantize_batch(const float* x, float* scales, int8_t* qx,
                               dim_t batch_size, dim_t depth, float shift = 0);

    template <typename T>
    static void dequantize_batch(const T* x, const float* scale, float* y,
                                 dim_t x_size, dim_t scale_size, float shift = 0);

    static void rescale_output(const int32_t* c,
                               const float* a_scales,
                               const float* b_scales,
                               const bool transpose_a,
                               const bool transpose_b,
                               float* y,
                               dim_t batch_size,
                               dim_t depth);

    template <typename T>
    static void transpose_2d(const T* a, const dim_t* dims, T* b);
    template <typename T>
    static void transpose_3d(const T* a, const dim_t* dims, const dim_t* perm, T* b);
    template <typename T>
    static void transpose_4d(const T* a, const dim_t* dims, const dim_t* perm, T* b);

    static void pow(const float* x, float* y, float power, dim_t size);
    static void exp(const float* x, float* y, dim_t size);
    static void log(const float* x, float* y, dim_t size);
    static void cos(const float* x, float* y, dim_t size);
    static void sin(const float* x, float* y, dim_t size);
    static void tanh(const float* x, float* y, dim_t size);
    static void relu(const float* x, float* y, dim_t size);
    static void gelu(const float* x, float* y, dim_t size);

    static void compute_u8_compensation(const int8_t* b,
                                        bool transpose_b,
                                        dim_t k,
                                        dim_t n,
                                        float alpha,
                                        int32_t* compensation);
    static bool prefer_u8s8s32_gemm();

    // If dest is not passed, returns the number of bytes required to store the packed data,
    // or 0 if packing is not supported.
    template <typename T>
    static dim_t gemm_pack_b(const T* b,
                             const bool transpose_b,
                             const dim_t k,
                             const dim_t n,
                             const float alpha,
                             T* dest = nullptr);

    template <typename In, typename Out>
    static void gemm(const In* a, const In* b,
                     bool a_is_packed, bool b_is_packed,
                     bool transpose_a, bool transpose_b,
                     dim_t m, dim_t n, dim_t k,
                     float alpha, float beta,
                     Out* c,
                     const Out* a_shift_compensation = nullptr);

    template <typename In, typename Out>
    static void gemm_batch(const In* a, const In* b,
                           bool transpose_a, bool transpose_b,
                           dim_t batch_size,
                           dim_t m, dim_t n, dim_t k,
                           float alpha, float beta,
                           Out* c);
  };

  template <Device D1, Device D2>
  struct cross_device_primitives {
    template <typename T>
    static void copy(const T* x, T* y, dim_t size);
  };

}
