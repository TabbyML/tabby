#pragma once

#include <algorithm>

namespace onmt {
  namespace compute {

    template <typename T>
    void fill(T* x, T a, size_t size) {
      std::fill_n(x, size, a);
    }

    template <typename T>
    void copy(const T* x, T* y, size_t size) {
      std::copy_n(x, size, y);
    }

    template <typename T>
    T sum(const T* array, size_t size) {
      return std::accumulate(array, array + size, static_cast<T>(0));
    }

    template <typename T>
    T mean(const T* array, size_t size) {
      return sum(array, size) / size;
    }

    template <typename T>
    size_t max_element(const T* array, size_t size) {
      return std::max_element(array, array + size) - array;
    }

    template <typename T>
    T max(const T* array, size_t size) {
      return array[max_element(array, size)];
    }

    template <typename T, typename I>
    void topk(const T* x, I* indices, size_t k, size_t size) {
      const auto comp = [&x](I i1, I i2) {
        return x[i1] > x[i2];
      };
      for (I i = 0; i < static_cast<I>(size); ++i)
        indices[i] = i;
      std::nth_element(indices, indices + k, indices + size, comp);
      std::sort(indices, indices + k, comp);
    }

    template <typename T>
    void add(T a, T* y, size_t size) {
      for (size_t i = 0; i < size; ++i)
        y[i] += a;
    }

    template <typename T>
    void add(const T* x, T* y, size_t size) {
      for (size_t i = 0; i < size; ++i)
        y[i] += x[i];
    }

    template <typename T>
    void sub(T a, T* y, size_t size) {
      T a_rev = -a;
      add(a_rev, y, size);
    }

    template <typename T>
    void mul(T a, T* y, size_t size) {
      for (size_t i = 0; i < size; ++i)
        y[i] *= a;
    }

    template <typename T>
    void mul(const T* x, T* y, size_t size) {
      for (size_t i = 0; i < size; ++i)
        y[i] *= x[i];
    }

    template <typename T>
    void mul(const T* a, const T* b, T* c, size_t size) {
      for (size_t i = 0; i < size; ++i)
        c[i] = a[i] * b[i];
    }

    template <typename T>
    void inv(const T* x, T* y, size_t size) {
      for (size_t i = 0; i < size; ++i)
        y[i] = static_cast<T>(1) / x[i];
    }

    template <typename In, typename Out>
    void quantize(const In* x, Out* y, size_t size, In scale, In shift) {
      for (size_t i = 0; i < size; ++i)
        y[i] = static_cast<Out>(x[i] * scale + shift);
    }

    template <typename In, typename Out>
    void unquantize(const In* x, Out* y, size_t size, Out scale, Out shift) {
      for (size_t i = 0; i < size; ++i)
        y[i] = (static_cast<Out>(x[i]) - shift) / scale;
    }

    template <typename T>
    void relu(T* x, size_t size) {
      for (size_t i = 0; i < size; ++i) {
        if (x[i] < static_cast<T>(0))
          x[i] = static_cast<T>(0);
      }
    }

    template <typename T>
    void relu(const T* x, T* y, size_t size) {
      for (size_t i = 0; i < size; ++i) {
        y[i] = x[i] > 0 ? x[i] : static_cast<T>(0);
      }
    }

    // Functions without generic implementation.
    template <typename T>
    void pow(const T* x, T* y, T power, size_t size);
    template <typename T>
    void exp(const T* x, T* y, size_t size);
    template <typename T>
    void cos(const T* x, T* y, size_t size);
    template <typename T>
    void sin(const T* x, T* y, size_t size);
    template <typename T>
    void tanh(const T* x, T* y, size_t size);

    template <typename T>
    void transpose_2d_inplace(T* a, size_t rows, size_t cols);
    template <typename T>
    void transpose_2d(const T* a, size_t rows, size_t cols, T* b);

    template <typename In, typename Out>
    void gemm(const In* a, const In* b,
              bool transpose_a, bool transpose_b,
              size_t m, size_t n, size_t k,
              In alpha, Out beta,
              Out* c);
    template <typename In, typename Out>
    void gemm_batch(const In* a, const In* b,
                    bool transpose_a, bool transpose_b,
                    size_t batch_size,
                    size_t m, size_t n, size_t k,
                    In alpha, Out beta,
                    Out* c) {
      for (size_t i = 0; i < batch_size; ++i) {
        const In* a_i = a + (i * m * k);
        const In* b_i = b + (i * k * n);
        Out* c_i = c + (i * m * n);

        gemm(a_i, b_i, transpose_a, transpose_b, m, n, k, alpha, beta, c_i);
      }
    }

  }
}
