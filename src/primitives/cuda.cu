#include "ctranslate2/primitives/primitives.h"

#include <cmath>
#include <type_traits>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cuda/helpers.h"
#include "type_dispatch.h"

namespace ctranslate2 {

  template<>
  template <typename T>
  T primitives<Device::CUDA>::deref(const T* x, dim_t index) {
    T val = T();
    cross_device_primitives<Device::CUDA, Device::CPU>::copy(x + index, &val, 1);
    return val;
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::fill(T* x, T a, dim_t size) {
    THRUST_CALL(thrust::fill, x, x + size, a);
  }
  template<>
  template <typename T>
  void primitives<Device::CUDA>::strided_fill(T* x, T a, dim_t inc_x, dim_t size) {
    auto it = thrust::make_permutation_iterator(
      x, thrust::make_transform_iterator(thrust::counting_iterator<dim_t>(0),
                                         thrust::placeholders::_1 * inc_x));
    THRUST_CALL(thrust::fill, it, it + size, a);
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::copy(const T* x, T* y, dim_t size) {
    CUDA_CHECK(cudaMemcpyAsync(y, x, size * sizeof (T),
                               cudaMemcpyDeviceToDevice, cuda::get_cuda_stream()));
  }

  template<>
  template <typename U, typename V>
  void primitives<Device::CUDA>::convert(const U* x, V* y, dim_t size) {
    THRUST_CALL(thrust::copy,
                cuda::device_cast(x), cuda::device_cast(x) + size, cuda::device_cast(y));
  }

  template void primitives<Device::CUDA>::convert(const float*, float16_t*, dim_t);
  template void primitives<Device::CUDA>::convert(const float16_t*, float*, dim_t);

  template<>
  template <typename T>
  T primitives<Device::CUDA>::sum(const T* array, dim_t size) {
    return T(THRUST_CALL(thrust::reduce,
                         cuda::device_cast(array),
                         cuda::device_cast(array) + size,
                         cuda::device_type<T>(),
                         cuda::plus<cuda::device_type<T>>()));
  }

  template<>
  template <typename T>
  dim_t primitives<Device::CUDA>::max_element(const T* array, dim_t size) {
    const auto* max = THRUST_CALL(thrust::max_element,
                                  cuda::device_cast(array),
                                  cuda::device_cast(array) + size,
                                  cuda::maximum<cuda::device_type<T>>());
    return static_cast<dim_t>(max - cuda::device_cast(array));
  }

  template<>
  template <typename T>
  T primitives<Device::CUDA>::max(const T* array, dim_t size) {
    return deref(array, max_element(array, size));
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::add(T a, const T* x, T* y, dim_t size) {
    using DeviceT = cuda::device_type<T>;
    cuda::unary_transform(x, y, size, cuda::bind_right<cuda::plus, DeviceT>(DeviceT(a)));
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::add(const T* a, const T* b, T* c, dim_t size) {
    cuda::binary_transform(a, b, c, size, cuda::plus<cuda::device_type<T>>());
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::add_batch_broadcast(const T* a, const T* b, T* c,
                                                     dim_t a_size, dim_t b_size) {
    cuda::binary_transform(a, b, c, b_size,
                           cuda::plus<cuda::device_type<T>>(),
                           cuda::repeat_vec<dim_t>(a_size));
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::add_depth_broadcast(const T* a, const T* b, T* c,
                                                     dim_t a_size, dim_t b_size) {
    cuda::binary_transform(a, b, c, b_size,
                           cuda::plus<cuda::device_type<T>>(),
                           cuda::repeat_vec_depth<dim_t>(b_size / a_size));
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::sub(const T* a, const T* b, T* c, dim_t size) {
    cuda::binary_transform(a, b, c, size, cuda::minus<cuda::device_type<T>>());
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::min(T a, const T* x, T* y, dim_t size) {
    using DeviceT = cuda::device_type<T>;
    cuda::unary_transform(x, y, size, cuda::bind_right<cuda::minimum, DeviceT>(DeviceT(a)));
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::min(const T* a, const T* b, T* c, dim_t size) {
    cuda::binary_transform(a, b, c, size, cuda::minimum<cuda::device_type<T>>());
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::max(T a, const T* x, T* y, dim_t size) {
    using DeviceT = cuda::device_type<T>;
    cuda::unary_transform(x, y, size, cuda::bind_right<cuda::maximum, DeviceT>(DeviceT(a)));
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::max(const T* a, const T* b, T* c, dim_t size) {
    cuda::binary_transform(a, b, c, size, cuda::maximum<cuda::device_type<T>>());
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::mul(T a, const T* x, T* y, dim_t size) {
    using DeviceT = cuda::device_type<T>;
    cuda::unary_transform(x, y, size, cuda::bind_right<cuda::multiplies, DeviceT>(DeviceT(a)));
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::mul(const T* a, const T* b, T* c, dim_t size) {
    cuda::binary_transform(a, b, c, size, cuda::multiplies<cuda::device_type<T>>());
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::mul_batch_broadcast(const T* a, const T* b, T* c,
                                                     dim_t a_size, dim_t b_size) {
    cuda::binary_transform(a, b, c, b_size,
                           cuda::multiplies<cuda::device_type<T>>(),
                           cuda::repeat_vec<dim_t>(a_size));
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::relu(const T* x, T* y, dim_t size) {
    max(T(0), x, y, size);
  }

  template void primitives<Device::CUDA>::relu(const float*, float*, dim_t);
  template void primitives<Device::CUDA>::relu(const float16_t*, float16_t*, dim_t);

  struct gelu_func {
    float _scale;
    gelu_func(float scale)
      : _scale(scale) {
    }
    __host__ __device__
    float operator()(float x) {
      return 0.5f * x * (1.f + tanhf(_scale * (x + 0.044715f * powf(x, 3.f))));
    }
  };

  template<>
  void primitives<Device::CUDA>::gelu(const float* x, float* y, dim_t size) {
    static const float pi = std::acos(-1.f);
    static const float scale = std::sqrt(2.f / pi);
    cuda::unary_transform(x, y, size, gelu_func(scale));
  }

  template <typename T>
  struct perm_indices_2d {
    T _rows, _cols;
    perm_indices_2d(T rows, T cols)
      : _rows(rows)
      , _cols(cols) {
    }
    __host__ __device__
    T operator()(const T i) const {
      const T i0 = i / _rows;
      const T i1 = i % _rows;
      return i1 * _cols + i0;
    }
  };

  template<>
  template <typename T>
  void primitives<Device::CUDA>::transpose_2d(const T* a, const dim_t* dims, T* b) {
    cuda::permute(a, b, dims[0] * dims[1], perm_indices_2d<dim_t>(dims[0], dims[1]));
  }

  template <typename T>
  struct perm_indices_3d {
    T _a_ps0, _a_ps1, _a_ps2; // Permuted strides of the original array.
    T _b_d0, _b_d1, _b_d2;    // Dimension of the permutated array.
    T _b_s0, _b_s1, _b_s2;    // Strides of the permutated array.
    perm_indices_3d(const T* dims, const T* perm) {
      const T a_stride[3] = {dims[1] * dims[2], dims[2], 1};
      _a_ps0 = a_stride[perm[0]];
      _a_ps1 = a_stride[perm[1]];
      _a_ps2 = a_stride[perm[2]];
      _b_d0 = dims[perm[0]];
      _b_d1 = dims[perm[1]];
      _b_d2 = dims[perm[2]];
      _b_s0 = _b_d1 * _b_d2;
      _b_s1 = _b_d2;
      _b_s2 = 1;
    }
    __host__ __device__
    T operator()(const T i) const {
      const T i0 = i / _b_s0;
      const T i1 = i / _b_s1 % _b_d1;
      const T i2 = i % _b_d2;
      return i0 * _a_ps0 + i1 * _a_ps1 + i2 * _a_ps2;
    }
  };

  template<>
  template <typename T>
  void primitives<Device::CUDA>::transpose_3d(const T* a,
                                              const dim_t* dims,
                                              const dim_t* perm,
                                              T* b) {
    cuda::permute(a, b, dims[0] * dims[1] * dims[2], perm_indices_3d<dim_t>(dims, perm));
  }

  template <typename T>
  struct perm_indices_4d {
    T _a_ps0, _a_ps1, _a_ps2, _a_ps3; // Permuted strides of the original array.
    T _b_d0, _b_d1, _b_d2, _b_d3;    // Dimension of the permutated array.
    T _b_s0, _b_s1, _b_s2, _b_s3;    // Strides of the permutated array.
    perm_indices_4d(const T* dims, const T* perm) {
      const T a_stride[4] = {dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1};
      _a_ps0 = a_stride[perm[0]];
      _a_ps1 = a_stride[perm[1]];
      _a_ps2 = a_stride[perm[2]];
      _a_ps3 = a_stride[perm[3]];
      _b_d0 = dims[perm[0]];
      _b_d1 = dims[perm[1]];
      _b_d2 = dims[perm[2]];
      _b_d3 = dims[perm[3]];
      _b_s0 = _b_d1 * _b_d2 * _b_d3;
      _b_s1 = _b_d2 * _b_d3;
      _b_s2 = _b_d3;
      _b_s3 = 1;
    }
    __host__ __device__
    T operator()(const T i) const {
      const T i0 = i / _b_s0;
      const T i1 = i / _b_s1 % _b_d1;
      const T i2 = i / _b_s2 % _b_d2;
      const T i3 = i % _b_d3;
      return i0 * _a_ps0 + i1 * _a_ps1 + i2 * _a_ps2 + i3 * _a_ps3;
    }
  };

  template <typename T>
  __global__ void transpose_0213(const T* in,
                                 const dim_t rows,
                                 const dim_t cols,
                                 const dim_t stride1,
                                 const dim_t stride2,
                                 T* out) {
    const dim_t stride = stride1 * stride2;
    for (dim_t j = blockIdx.x; j < rows; j += gridDim.x) {
      const dim_t z = j / stride;
      const dim_t y = (j % stride) / stride1;
      const dim_t x = (j % stride) % stride1;
      const dim_t j2 = z * stride + x * stride2 + y;

      const T* row_in = in + j2 * cols;
      T* row_out = out + j * cols;

      for (dim_t i = threadIdx.x; i < cols; i += blockDim.x) {
        row_out[i] = row_in[i];
      }
    }
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::transpose_4d(const T* a,
                                              const dim_t* dims,
                                              const dim_t* perm,
                                              T* b) {
    if (perm[0] == 0 && perm[1] == 2 && perm[2] == 1 && perm[3] == 3) {
      // Optimize the permutation used in multi-head attention.
      const dim_t rows = dims[0] * dims[1] * dims[2];
      const dim_t cols = dims[3];
      const dim_t blocks = std::min(rows, cuda::max_blocks);
      const dim_t threads = std::min(cols, cuda::max_threads);
      transpose_0213<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(a,
                                                                      rows,
                                                                      cols,
                                                                      dims[1],
                                                                      dims[2],
                                                                      b);
      return;
    }

    cuda::permute(a, b, dims[0] * dims[1] * dims[2] * dims[3], perm_indices_4d<dim_t>(dims, perm));
  }

  template<>
  template<>
  void primitives<Device::CUDA>::gemm(const float* a, const float* b,
                                      bool, bool,
                                      bool transpose_a, bool transpose_b,
                                      dim_t m, dim_t n, dim_t k,
                                      float alpha, float beta,
                                      float* c,
                                      const float*) {
    // Memo: cuBLAS assumes column-major storage.

    const int lda = transpose_a ? m : k;
    const int ldb = transpose_b ? k : n;
    const int ldc = n;

    const cublasOperation_t transa = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t transb = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    CUBLAS_CHECK(cublasSgemm(cuda::get_cublas_handle(),
                             transb, transa,
                             n, m, k,
                             &alpha,
                             b, ldb,
                             a, lda,
                             &beta,
                             c, ldc));
  }

  template<>
  template<>
  void primitives<Device::CUDA>::gemm(const float16_t* a, const float16_t* b,
                                      bool, bool,
                                      bool transpose_a, bool transpose_b,
                                      dim_t m, dim_t n, dim_t k,
                                      float alpha, float beta,
                                      float16_t* c,
                                      const float16_t*) {
    const int lda = transpose_a ? m : k;
    const int ldb = transpose_b ? k : n;
    const int ldc = n;

    const cublasOperation_t transa = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t transb = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    const __half alpha_h = alpha;
    const __half beta_h = beta;

    // cuBLAS assumes column-major storage, so swap a and b accordingly.
    CUBLAS_CHECK(cublasGemmEx(cuda::get_cublas_handle(),
                              transb, transa,
                              n, m, k,
                              &alpha_h,
                              b, CUDA_R_16F, ldb,
                              a, CUDA_R_16F, lda,
                              &beta_h,
                              c, CUDA_R_16F, ldc,
                              CUDA_R_16F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }

  template<>
  template<>
  void primitives<Device::CUDA>::gemm(const int8_t* a, const int8_t* b,
                                      bool, bool,
                                      bool transpose_a, bool transpose_b,
                                      dim_t m, dim_t n, dim_t k,
                                      float alpha, float beta,
                                      int32_t* c,
                                      const int32_t*) {
    const int lda = transpose_a ? m : k;
    const int ldb = transpose_b ? k : n;
    const int ldc = n;

    const cublasOperation_t transa = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t transb = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    int32_t alpha_i = alpha;
    int32_t beta_i = beta;

    // cuBLAS assumes column-major storage, so swap a and b accordingly.
    CUBLAS_CHECK(cublasGemmEx(cuda::get_cublas_handle(),
                              transb, transa,
                              n, m, k,
                              &alpha_i,
                              b, CUDA_R_8I, ldb,
                              a, CUDA_R_8I, lda,
                              &beta_i,
                              c, CUDA_R_32I, ldc,
                              CUDA_R_32I,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }

  template<>
  template<>
  void primitives<Device::CUDA>::gemm_batch(const float* a, const float* b,
                                            bool transpose_a, bool transpose_b,
                                            dim_t batch_size,
                                            dim_t m, dim_t n, dim_t k,
                                            float alpha, float beta,
                                            float* c) {
    // Memo: cuBLAS assumes column-major storage.

    const int lda = transpose_a ? m : k;
    const int ldb = transpose_b ? k : n;
    const int ldc = n;

    const long long int stridea = m * k;
    const long long int strideb = k * n;
    const long long int stridec = m * n;

    const cublasOperation_t transa = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t transb = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    CUBLAS_CHECK(cublasSgemmStridedBatched(cuda::get_cublas_handle(),
                                           transb, transa,
                                           n, m, k,
                                           &alpha,
                                           b, ldb, strideb,
                                           a, lda, stridea,
                                           &beta,
                                           c, ldc, stridec,
                                           batch_size));
  }

  template<>
  template<>
  void primitives<Device::CUDA>::gemm_batch(const float16_t* a, const float16_t* b,
                                            bool transpose_a, bool transpose_b,
                                            dim_t batch_size,
                                            dim_t m, dim_t n, dim_t k,
                                            float alpha, float beta,
                                            float16_t* c) {
    const int lda = transpose_a ? m : k;
    const int ldb = transpose_b ? k : n;
    const int ldc = n;

    const long long int stridea = m * k;
    const long long int strideb = k * n;
    const long long int stridec = m * n;

    const cublasOperation_t transa = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t transb = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    const __half alpha_h = alpha;
    const __half beta_h = beta;

    // cuBLAS assumes column-major storage, so swap a and b accordingly.
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(cuda::get_cublas_handle(),
                                            transb, transa,
                                            n, m, k,
                                            &alpha_h,
                                            b, CUDA_R_16F, ldb, strideb,
                                            a, CUDA_R_16F, lda, stridea,
                                            &beta_h,
                                            c, CUDA_R_16F, ldc, stridec,
                                            batch_size,
                                            CUDA_R_16F,
                                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }

  struct exp_func {
    __host__ __device__
    float operator()(float x) { return expf(x); }
  };

  template<>
  void primitives<Device::CUDA>::exp(const float* x, float* y, dim_t size) {
    cuda::unary_transform(x, y, size, exp_func());
  }

  struct log_func {
    __host__ __device__
    float operator()(float x) { return logf(x); }
  };

  template<>
  void primitives<Device::CUDA>::log(const float* x, float* y, dim_t size) {
    cuda::unary_transform(x, y, size, log_func());
  }


  template<>
  template <typename T>
  void cross_device_primitives<Device::CPU, Device::CUDA>::copy(const T* x, T* y, dim_t size) {
    CUDA_CHECK(cudaMemcpyAsync(y, x, size * sizeof (T), cudaMemcpyHostToDevice, cuda::get_cuda_stream()));
  }

  template<>
  template <typename T>
  void cross_device_primitives<Device::CUDA, Device::CPU>::copy(const T* x, T* y, dim_t size) {
    CUDA_CHECK(cudaMemcpyAsync(y, x, size * sizeof (T), cudaMemcpyDeviceToHost, cuda::get_cuda_stream()));
  }

#define DECLARE_IMPL(T)                                                 \
  template T                                                            \
  primitives<Device::CUDA>::deref(const T* x, dim_t index);             \
  template void                                                         \
  primitives<Device::CUDA>::fill(T* x, T a, dim_t size);                \
  template void                                                         \
  primitives<Device::CUDA>::strided_fill(T* x, T a, dim_t inc_x, dim_t size); \
  template void                                                         \
  primitives<Device::CUDA>::copy<T>(const T* x, T* y, dim_t size);      \
  template T                                                            \
  primitives<Device::CUDA>::sum(const T* array, dim_t size);            \
  template dim_t                                                        \
  primitives<Device::CUDA>::max_element(const T* array, dim_t size);    \
  template T                                                            \
  primitives<Device::CUDA>::max(const T* array, dim_t size);            \
  template void                                                         \
  primitives<Device::CUDA>::add(T a, const T* x, T* y, dim_t size);     \
  template void                                                         \
  primitives<Device::CUDA>::add(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CUDA>::add_batch_broadcast(const T* a, const T* b, \
                                                T* c, dim_t a_size, dim_t b_size); \
  template void                                                         \
  primitives<Device::CUDA>::add_depth_broadcast(const T* a, const T* b, \
                                                T* c, dim_t a_size, dim_t b_size); \
  template void                                                         \
  primitives<Device::CUDA>::sub(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CUDA>::min(T a, const T* x, T* y, dim_t size);      \
  template void                                                         \
  primitives<Device::CUDA>::min(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CUDA>::max(T a, const T* x, T* y, dim_t size);     \
  template void                                                         \
  primitives<Device::CUDA>::max(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CUDA>::mul(T a, const T* x, T* y, dim_t size);     \
  template void                                                         \
  primitives<Device::CUDA>::mul(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CUDA>::mul_batch_broadcast(const T* a, const T* b, \
                                                T* c, dim_t a_size, dim_t b_size); \
  template void                                                         \
  primitives<Device::CUDA>::transpose_2d(const T* a,                    \
                                         const dim_t* dims,             \
                                         T* b);                         \
  template void                                                         \
  primitives<Device::CUDA>::transpose_3d(const T* a,                    \
                                         const dim_t* dims,             \
                                         const dim_t* perm,             \
                                         T* b);                         \
  template void                                                         \
  primitives<Device::CUDA>::transpose_4d(const T* a,                    \
                                         const dim_t* dims,             \
                                         const dim_t* perm,             \
                                         T* b);                         \
  template void                                                         \
  cross_device_primitives<Device::CPU, Device::CUDA>::copy<T>(const T*, T*, dim_t); \
  template void                                                         \
  cross_device_primitives<Device::CUDA, Device::CPU>::copy<T>(const T*, T*, dim_t);

  DECLARE_ALL_TYPES(DECLARE_IMPL)

}

