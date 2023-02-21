#include "ctranslate2/primitives.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_ptr.h>

#include "cuda/helpers.h"
#include "type_dispatch.h"

namespace ctranslate2 {

  template<>
  template <typename T>
  T primitives<Device::CUDA>::at(const T* x, dim_t index) {
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
      x, thrust::make_transform_iterator(thrust::counting_iterator<cuda::index_t>(0),
                                         thrust::placeholders::_1 * inc_x));
    THRUST_CALL(thrust::fill, it, it + size, a);
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::indexed_fill(T* x, T a, const int32_t* indices, dim_t num_indices) {
    auto element_it = thrust::device_pointer_cast(cuda::device_cast(x));
    auto index_it = thrust::device_pointer_cast(indices);
    auto it = thrust::make_permutation_iterator(element_it, index_it);
    THRUST_CALL(thrust::fill, it, it + num_indices, cuda::device_type<T>(a));
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
    return T(THRUST_CALL(thrust::reduce,
                         cuda::device_cast(array),
                         cuda::device_cast(array) + size,
                         cuda::device_type<T>(std::numeric_limits<T>::lowest()),
                         cuda::maximum<cuda::device_type<T>>()));
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
                           cuda::repeat_vec<cuda::index_t>(a_size));
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::add_depth_broadcast(const T* a, const T* b, T* c,
                                                     dim_t a_size, dim_t b_size) {
    cuda::binary_transform(a, b, c, b_size,
                           cuda::plus<cuda::device_type<T>>(),
                           cuda::repeat_vec_depth<cuda::index_t>(b_size / a_size));
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
                           cuda::repeat_vec<cuda::index_t>(a_size));
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::relu(const T* x, T* y, dim_t size) {
    cuda::unary_transform(x, y, size, cuda::relu_func<cuda::device_type<T>>());
  }

  template void primitives<Device::CUDA>::relu(const float*, float*, dim_t);
  template void primitives<Device::CUDA>::relu(const float16_t*, float16_t*, dim_t);

  template<>
  template <typename T>
  void primitives<Device::CUDA>::gelu(const T* x, T* y, dim_t size) {
    cuda::unary_transform(x, y, size, cuda::gelu_func<cuda::device_type<T>>());
  }

  template void primitives<Device::CUDA>::gelu(const float*, float*, dim_t);
  template void primitives<Device::CUDA>::gelu(const float16_t*, float16_t*, dim_t);

  template<>
  template <typename T>
  void primitives<Device::CUDA>::gelu_tanh(const T* x, T* y, dim_t size) {
    cuda::unary_transform(x, y, size, cuda::gelu_tanh_func<cuda::device_type<T>>());
  }

  template void primitives<Device::CUDA>::gelu_tanh(const float*, float*, dim_t);
  template void primitives<Device::CUDA>::gelu_tanh(const float16_t*, float16_t*, dim_t);

  template<>
  template <typename T>
  void primitives<Device::CUDA>::gelu_sigmoid(const T* x, T* y, dim_t size) {
    cuda::unary_transform(x, y, size, cuda::gelu_sigmoid_func<cuda::device_type<T>>());
  }

  template void primitives<Device::CUDA>::gelu_sigmoid(const float*, float*, dim_t);
  template void primitives<Device::CUDA>::gelu_sigmoid(const float16_t*, float16_t*, dim_t);

  template<>
  template <typename T>
  void primitives<Device::CUDA>::swish(const T* x, T* y, dim_t size) {
    cuda::unary_transform(x, y, size, cuda::swish_func<cuda::device_type<T>>());
  }

  template void primitives<Device::CUDA>::swish(const float*, float*, dim_t);
  template void primitives<Device::CUDA>::swish(const float16_t*, float16_t*, dim_t);

  template <typename T>
  struct perm_indices_2d {
    T _rows, _cols;
    perm_indices_2d(T rows, T cols)
      : _rows(rows)
      , _cols(cols) {
    }
    __device__
    T operator()(const T i) const {
      const T i0 = i / _rows;
      const T i1 = i % _rows;
      return i1 * _cols + i0;
    }
  };

  template <typename T>
  __global__ void penalize_previous_tokens_kernel(T* scores,
                                                  const T* previous_scores,
                                                  const int32_t* previous_ids,
                                                  float penalty,
                                                  cuda::index_t batch_size,
                                                  cuda::index_t length,
                                                  cuda::index_t vocabulary_size) {
    for (cuda::index_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < batch_size * length;
         i += blockDim.x * gridDim.x) {
      const cuda::index_t write_index = (i / length) * vocabulary_size + previous_ids[i];
      const float score = previous_scores[i];
      scores[write_index] = (score < 0.f ? score * penalty : score / penalty);
    }
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::penalize_previous_tokens(T* scores,
                                                          const T* previous_scores,
                                                          const int32_t* previous_ids,
                                                          T penalty,
                                                          dim_t batch_size,
                                                          dim_t length,
                                                          dim_t vocabulary_size) {
    dim3 block(32);
    dim3 grid((batch_size * length + block.x - 1) / block.x);
    penalize_previous_tokens_kernel<<<grid, block, 0, cuda::get_cuda_stream()>>>(
      cuda::device_cast(scores),
      cuda::device_cast(previous_scores),
      previous_ids,
      penalty,
      batch_size,
      length,
      vocabulary_size);
  }

  __global__ void prepare_length_mask_kernel(const int32_t* lengths,
                                             cuda::index_t num_heads,
                                             cuda::index_t num_queries,
                                             bool mask_future,
                                             int32_t* mask) {
    const auto length = lengths[blockIdx.x];
    mask += blockIdx.x * num_heads * num_queries;
    for (cuda::index_t i = threadIdx.x; i < num_heads * num_queries; i += blockDim.x)
      mask[i] = (mask_future ? min(length, int32_t((i % num_queries) + 1)) : length);
  }

  template<>
  void primitives<Device::CUDA>::prepare_length_mask(const int32_t* lengths,
                                                     dim_t batch_size,
                                                     dim_t num_heads,
                                                     dim_t num_queries,
                                                     bool mask_future,
                                                     int32_t* mask) {
    const dim_t blocks = std::min(batch_size, cuda::max_blocks);
    const dim_t threads = std::min(num_heads * num_queries, cuda::max_threads);
    prepare_length_mask_kernel<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(lengths,
                                                                                num_heads,
                                                                                num_queries,
                                                                                mask_future,
                                                                                mask);
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::transpose_2d(const T* a, const dim_t* dims, T* b) {
    cuda::permute(a, b, dims[0] * dims[1], perm_indices_2d<cuda::index_t>(dims[0], dims[1]));
  }

  template <typename T>
  struct perm_indices_3d {
    T _a_ps0, _a_ps1, _a_ps2; // Permuted strides of the original array.
    T _b_d0, _b_d1, _b_d2;    // Dimension of the permutated array.
    T _b_s0, _b_s1, _b_s2;    // Strides of the permutated array.
    perm_indices_3d(const dim_t* dims, const dim_t* perm) {
      const dim_t a_stride[3] = {dims[1] * dims[2], dims[2], 1};
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
    __device__
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
    cuda::permute(a, b, dims[0] * dims[1] * dims[2], perm_indices_3d<cuda::index_t>(dims, perm));
  }

  template <typename T>
  struct perm_indices_4d {
    T _a_ps0, _a_ps1, _a_ps2, _a_ps3; // Permuted strides of the original array.
    T _b_d0, _b_d1, _b_d2, _b_d3;    // Dimension of the permutated array.
    T _b_s0, _b_s1, _b_s2, _b_s3;    // Strides of the permutated array.
    perm_indices_4d(const dim_t* dims, const dim_t* perm) {
      const dim_t a_stride[4] = {dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1};
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
    __device__
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
                                 const cuda::index_t cols,
                                 const cuda::index_t stride1,
                                 const cuda::index_t stride2,
                                 T* out) {
    const cuda::index_t stride = stride1 * stride2;
    const cuda::index_t j = blockIdx.x;
    const cuda::index_t z = j / stride;
    const cuda::index_t y = (j % stride) / stride1;
    const cuda::index_t x = (j % stride) % stride1;
    const cuda::index_t j2 = z * stride + x * stride2 + y;

    const T* row_in = in + j2 * cols;
    T* row_out = out + j * cols;

    for (cuda::index_t i = threadIdx.x; i < cols; i += blockDim.x) {
      row_out[i] = row_in[i];
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
      const dim_t blocks = std::min(rows, cuda::max_blocks);

      if ((dims[3] * sizeof (T)) % sizeof(uint4) == 0) {
        const dim_t cols = (dims[3] * sizeof (T)) / sizeof (uint4);
        const dim_t threads = std::min(cols, cuda::max_threads);
        transpose_0213<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
          reinterpret_cast<const uint4*>(a),
          cols,
          dims[1],
          dims[2],
          reinterpret_cast<uint4*>(b));

      } else {
        const dim_t cols = dims[3];
        const dim_t threads = std::min(cols, cuda::max_threads);
        transpose_0213<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(a,
                                                                        cols,
                                                                        dims[1],
                                                                        dims[2],
                                                                        b);
      }

      return;
    }

    cuda::permute(a, b, dims[0] * dims[1] * dims[2] * dims[3],
                  perm_indices_4d<cuda::index_t>(dims, perm));
  }

  template<>
  template<>
  void primitives<Device::CUDA>::gemm(bool, bool,
                                      bool transpose_a, bool transpose_b,
                                      dim_t m, dim_t n, dim_t k,
                                      float alpha,
                                      const float* a, dim_t lda,
                                      const float* b, dim_t ldb,
                                      float beta,
                                      float* c, dim_t ldc,
                                      const float*) {
    // cuBLAS assumes column-major storage, so swap a and b accordingly.
    CUBLAS_CHECK(cublasSgemm(cuda::get_cublas_handle(),
                             transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,
                             transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                             n, m, k,
                             &alpha,
                             b, ldb,
                             a, lda,
                             &beta,
                             c, ldc));
  }

  template<>
  template<>
  void primitives<Device::CUDA>::gemm(bool, bool,
                                      bool transpose_a, bool transpose_b,
                                      dim_t m, dim_t n, dim_t k,
                                      float alpha,
                                      const float16_t* a, dim_t lda,
                                      const float16_t* b, dim_t ldb,
                                      float beta,
                                      float16_t* c, dim_t ldc,
                                      const float16_t*) {
    const __half alpha_h = alpha;
    const __half beta_h = beta;

    const void* alpha_ptr = &alpha_h;
    const void* beta_ptr = &beta_h;
    cudaDataType_t compute_type = CUDA_R_16F;

    if (!cuda::use_true_fp16_gemm()) {
      alpha_ptr = &alpha;
      beta_ptr = &beta;
      compute_type = CUDA_R_32F;
    }

    // cuBLAS assumes column-major storage, so swap a and b accordingly.
    CUBLAS_CHECK(cublasGemmEx(cuda::get_cublas_handle(),
                              transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,
                              transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                              n, m, k,
                              alpha_ptr,
                              b, CUDA_R_16F, ldb,
                              a, CUDA_R_16F, lda,
                              beta_ptr,
                              c, CUDA_R_16F, ldc,
                              compute_type,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }

  template<>
  template<>
  void primitives<Device::CUDA>::gemm(bool, bool,
                                      bool transpose_a, bool transpose_b,
                                      dim_t m, dim_t n, dim_t k,
                                      float alpha,
                                      const int8_t* a, dim_t lda,
                                      const int8_t* b, dim_t ldb,
                                      float beta,
                                      int32_t* c, dim_t ldc,
                                      const int32_t*) {
    int32_t alpha_i = alpha;
    int32_t beta_i = beta;

    // cuBLAS assumes column-major storage, so swap a and b accordingly.
    CUBLAS_CHECK(cublasGemmEx(cuda::get_cublas_handle(),
                              transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,
                              transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N,
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
  void primitives<Device::CUDA>::gemm_batch_strided(bool transpose_a, bool transpose_b,
                                                    dim_t m, dim_t n, dim_t k,
                                                    float alpha,
                                                    const float* a, dim_t lda, dim_t stridea,
                                                    const float* b, dim_t ldb, dim_t strideb,
                                                    float beta,
                                                    float* c, dim_t ldc, dim_t stridec,
                                                    dim_t batch_size) {
    // cuBLAS assumes column-major storage, so swap a and b accordingly.
    CUBLAS_CHECK(cublasSgemmStridedBatched(cuda::get_cublas_handle(),
                                           transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,
                                           transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N,
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
  void primitives<Device::CUDA>::gemm_batch_strided(bool transpose_a, bool transpose_b,
                                                    dim_t m, dim_t n, dim_t k,
                                                    float alpha,
                                                    const float16_t* a, dim_t lda, dim_t stridea,
                                                    const float16_t* b, dim_t ldb, dim_t strideb,
                                                    float beta,
                                                    float16_t* c, dim_t ldc, dim_t stridec,
                                                    dim_t batch_size) {
    const __half alpha_h = alpha;
    const __half beta_h = beta;

    const void* alpha_ptr = &alpha_h;
    const void* beta_ptr = &beta_h;
    cudaDataType_t compute_type = CUDA_R_16F;

    if (!cuda::use_true_fp16_gemm()) {
      alpha_ptr = &alpha;
      beta_ptr = &beta;
      compute_type = CUDA_R_32F;
    }

    // cuBLAS assumes column-major storage, so swap a and b accordingly.
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(cuda::get_cublas_handle(),
                                            transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,
                                            transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                                            n, m, k,
                                            alpha_ptr,
                                            b, CUDA_R_16F, ldb, strideb,
                                            a, CUDA_R_16F, lda, stridea,
                                            beta_ptr,
                                            c, CUDA_R_16F, ldc, stridec,
                                            batch_size,
                                            compute_type,
                                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }

  template <typename T>
  class exp_minus_max_func {
  private:
    const float _max_value;

  public:
    exp_minus_max_func(const float max_value)
      : _max_value(max_value)
    {
    }

    __device__ float operator()(T x) {
      return expf(float(x) - _max_value);
    }
  };

  template<>
  template <typename T>
  float primitives<Device::CUDA>::logsumexp(const T* x, dim_t size) {
    const float max_value = max(x, size);

    auto exp_it = thrust::make_transform_iterator(
      thrust::device_pointer_cast(cuda::device_cast(x)),
      exp_minus_max_func<cuda::device_type<T>>(max_value));

    const float exp_sum = THRUST_CALL(thrust::reduce, exp_it, exp_it + size);
    return std::log(exp_sum) + max_value;
  }

  template float primitives<Device::CUDA>::logsumexp(const float*, dim_t);
  template float primitives<Device::CUDA>::logsumexp(const float16_t*, dim_t);

  struct tanh_func {
    __device__ float operator()(float x) {
      return tanhf(x);
    }
  };

  template<>
  template <typename T>
  void primitives<Device::CUDA>::tanh(const T* x, T* y, dim_t size) {
    cuda::unary_transform(cuda::device_cast(x), cuda::device_cast(y), size, tanh_func());
  }

  template void primitives<Device::CUDA>::tanh(const float*, float*, dim_t);
  template void primitives<Device::CUDA>::tanh(const float16_t*, float16_t*, dim_t);

  struct exp_func {
    __device__
    float operator()(float x) { return expf(x); }
  };

  template<>
  void primitives<Device::CUDA>::exp(const float* x, float* y, dim_t size) {
    cuda::unary_transform(x, y, size, exp_func());
  }

  struct log_func {
    __device__
    float operator()(float x) { return logf(x); }
  };

  template<>
  template<>
  void primitives<Device::CUDA>::log(const float* x, float* y, dim_t size) {
    cuda::unary_transform(x, y, size, log_func());
  }

#if CUDA_CAN_USE_HALF
  struct hlog_func {
    __device__
    __half operator()(__half x) { return hlog(x); }
  };
#else
  struct hlog_func {
    __device__
    __half operator()(__half x) { return __half(logf(float(x))); }
  };
#endif

  template<>
  template<>
  void primitives<Device::CUDA>::log(const float16_t* x, float16_t* y, dim_t size) {
    cuda::unary_transform(x, y, size, hlog_func());
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
  primitives<Device::CUDA>::at(const T* x, dim_t index);                \
  template void                                                         \
  primitives<Device::CUDA>::fill(T* x, T a, dim_t size);                \
  template void                                                         \
  primitives<Device::CUDA>::strided_fill(T* x, T a, dim_t inc_x, dim_t size); \
  template void                                                         \
  primitives<Device::CUDA>::indexed_fill(T*, T, const int32_t*, dim_t); \
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
  primitives<Device::CUDA>::penalize_previous_tokens(T*,                \
                                                     const T*,          \
                                                     const int32_t*,    \
                                                     T,                 \
                                                     dim_t,             \
                                                     dim_t,             \
                                                     dim_t);            \
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
