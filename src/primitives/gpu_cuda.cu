#include "ctranslate2/primitives/gpu_cuda.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>

#include "ctranslate2/types.h"
#include "ctranslate2/cuda/utils.h"

namespace ctranslate2 {

  template <typename T, typename UnaryFunction>
  void unary_transform(const T* x, T* y, size_t size, UnaryFunction op) {
    thrust::transform(thrust::cuda::par.on(cuda::get_cuda_stream()), x, x + size, y, op);
  }

  template <typename T, typename BinaryFunction>
  void binary_transform(const T* a, const T* b, T* c, size_t size, BinaryFunction op) {
    thrust::transform(thrust::cuda::par.on(cuda::get_cuda_stream()), a, a + size, b, c, op);
  }


  template<>
  void* primitives<Device::CUDA>::alloc_data(size_t size) {
    void* data = nullptr;
    CUDA_CHECK(cudaMalloc(&data, size));
    return data;
  }

  template<>
  void primitives<Device::CUDA>::free_data(void* data) {
    CUDA_CHECK(cudaFree(data));
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::fill(T* x, T a, size_t size) {
    thrust::fill_n(thrust::cuda::par.on(cuda::get_cuda_stream()), x, size, a);
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::copy(const T* x, T* y, size_t size) {
    CUDA_CHECK(cudaMemcpyAsync(y, x, size * sizeof (T),
                               cudaMemcpyDeviceToDevice, cuda::get_cuda_stream()));
  }

  template<>
  template <typename T>
  T primitives<Device::CUDA>::sum(const T* array, size_t size) {
    return thrust::reduce(thrust::cuda::par.on(cuda::get_cuda_stream()), array, array + size);
  }

  template<>
  template <typename T>
  size_t primitives<Device::CUDA>::max_element(const T* array, size_t size) {
    const auto* max = thrust::max_element(thrust::cuda::par.on(cuda::get_cuda_stream()),
                                          array, array + size);
    return static_cast<size_t>(max - array);
  }

  template<>
  template <typename T>
  T primitives<Device::CUDA>::max(const T* array, size_t size) {
    thrust::device_ptr<const T> array_ptr(array);
    return *thrust::max_element(thrust::cuda::par.on(cuda::get_cuda_stream()),
                                array_ptr, array_ptr + size);
  }

  template<>
  template <typename T, typename I>
  void primitives<Device::CUDA>::topk(const T* x, T* val, I* ind, size_t k, size_t size) {
    static thread_local T* keys = nullptr;
    static thread_local I* values = nullptr;
    static thread_local size_t alloc_size = 0;

    if (size > alloc_size) {
      CUDA_CHECK(cudaFree(keys));
      CUDA_CHECK(cudaMalloc(&keys, size * sizeof (T)));
      CUDA_CHECK(cudaFree(values));
      CUDA_CHECK(cudaMalloc(&values, size * sizeof (I)));
      alloc_size = size;
    }

    copy(x, keys, size);
    thrust::sequence(thrust::cuda::par.on(cuda::get_cuda_stream()), values, values + size);
    thrust::sort_by_key(thrust::cuda::par.on(cuda::get_cuda_stream()),
                        keys, keys + size, values, thrust::greater<T>());
    copy(keys, val, k);
    copy(values, ind, k);
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::add(T a, const T* x, T* y, size_t size) {
    unary_transform(x, y, size, thrust::placeholders::_1 + a);
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::add(const T* a, const T* b, T* c, size_t size) {
    binary_transform(a, b, c, size, thrust::plus<T>());
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::sub(const T* a, const T* b, T* c, size_t size) {
    binary_transform(a, b, c, size, thrust::minus<T>());
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::mul(T a, const T* x, T* y, size_t size) {
    unary_transform(x, y, size, thrust::placeholders::_1 * a);
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::mul(const T* a, const T* b, T* c, size_t size) {
    binary_transform(a, b, c, size, thrust::multiplies<T>());
  }

  struct relu_func : public thrust::unary_function<float, float> {
    __host__ __device__
    float operator()(float x) { return fmaxf(x, 0); }
  };

  template<>
  template<>
  void primitives<Device::CUDA>::relu(const float* x, float* y, size_t size) {
    unary_transform(x, y, size, relu_func());
  }

  template<>
  template<>
  void primitives<Device::CUDA>::gemm(const float* a, const float* b,
                                      bool transpose_a, bool transpose_b,
                                      size_t m, size_t n, size_t k,
                                      float alpha, float beta,
                                      float* c) {
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
  void primitives<Device::CUDA>::gemm_batch(const float* a, const float* b,
                                            bool transpose_a, bool transpose_b,
                                            size_t batch_size,
                                            size_t m, size_t n, size_t k,
                                            float alpha, float beta,
                                            float* c) {
    // Memo: cuBLAS assumes column-major storage.

    const int lda = transpose_a ? m : k;
    const int ldb = transpose_b ? k : n;
    const int ldc = n;

    const cublasOperation_t transa = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t transb = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    const float** a_array = new const float*[batch_size];
    const float** b_array = new const float*[batch_size];
    float** c_array = new float*[batch_size];

    for (size_t i = 0; i < batch_size; ++i) {
      a_array[i] = a + (i * m * k);
      b_array[i] = b + (i * k * n);
      c_array[i] = c + (i * m * n);
    }

    static thread_local const float** a_array_device = nullptr;
    static thread_local const float** b_array_device = nullptr;
    static thread_local float** c_array_device = nullptr;
    static thread_local size_t alloc_size = 0;

    const size_t array_size = batch_size * sizeof (float*);

    if (array_size > alloc_size) {
      CUDA_CHECK(cudaFree(a_array_device));
      CUDA_CHECK(cudaFree(b_array_device));
      CUDA_CHECK(cudaFree(c_array_device));
      CUDA_CHECK(cudaMalloc(&a_array_device, array_size));
      CUDA_CHECK(cudaMalloc(&b_array_device, array_size));
      CUDA_CHECK(cudaMalloc(&c_array_device, array_size));
      alloc_size = array_size;
    }

    cross_device_primitives<Device::CPU, Device::CUDA>::copy(a_array, a_array_device, batch_size);
    cross_device_primitives<Device::CPU, Device::CUDA>::copy(b_array, b_array_device, batch_size);
    cross_device_primitives<Device::CPU, Device::CUDA>::copy(c_array, c_array_device, batch_size);

    delete [] a_array;
    delete [] b_array;
    delete [] c_array;

    CUBLAS_CHECK(cublasSgemmBatched(cuda::get_cublas_handle(),
                                    transb, transa,
                                    n, m, k,
                                    &alpha,
                                    b_array_device, ldb,
                                    a_array_device, lda,
                                    &beta,
                                    c_array_device, ldc,
                                    batch_size));
  }

  struct exp_func : public thrust::unary_function<float, float> {
    __host__ __device__
    float operator()(float x) { return expf(x); }
  };

  template<>
  template<>
  void primitives<Device::CUDA>::exp(const float* x, float* y, size_t size) {
    unary_transform(x, y, size, exp_func());
  }


  template<>
  template <typename T>
  void cross_device_primitives<Device::CPU, Device::CUDA>::copy(const T* x, T* y, size_t size) {
    CUDA_CHECK(cudaMemcpyAsync(y, x, size * sizeof (T), cudaMemcpyHostToDevice, cuda::get_cuda_stream()));
  }

  template<>
  template <typename T>
  void cross_device_primitives<Device::CUDA, Device::CPU>::copy(const T* x, T* y, size_t size) {
    CUDA_CHECK(cudaMemcpyAsync(y, x, size * sizeof (T), cudaMemcpyDeviceToHost, cuda::get_cuda_stream()));
  }

#define DECLARE_IMPL(T)                                                 \
  template void                                                         \
  primitives<Device::CUDA>::fill(T* x, T a, size_t size);               \
  template void                                                         \
  primitives<Device::CUDA>::copy<T>(const T* x, T* y, size_t size);     \
  template T                                                            \
  primitives<Device::CUDA>::sum(const T* array, size_t size);           \
  template size_t                                                       \
  primitives<Device::CUDA>::max_element(const T* array, size_t size);   \
  template T                                                            \
  primitives<Device::CUDA>::max(const T* array, size_t size);           \
  template void                                                         \
  primitives<Device::CUDA>::topk(const T* x, T* values, int* indices, size_t k, size_t size); \
  template void                                                         \
  primitives<Device::CUDA>::add(T a, const T* x, T* y, size_t size);    \
  template void                                                         \
  primitives<Device::CUDA>::add(const T* a, const T* b, T* c, size_t size); \
  template void                                                         \
  primitives<Device::CUDA>::sub(const T* a, const T* b, T* c, size_t size); \
  template void                                                         \
  primitives<Device::CUDA>::mul(T a, const T* x, T* y, size_t size);    \
  template void                                                         \
  primitives<Device::CUDA>::mul(const T* a, const T* b, T* c, size_t size); \
  template void                                                         \
  cross_device_primitives<Device::CPU, Device::CUDA>::copy<T>(const T*, T*, size_t); \
  template void                                                         \
  cross_device_primitives<Device::CUDA, Device::CPU>::copy<T>(const T*, T*, size_t);

  DECLARE_ALL_TYPES(DECLARE_IMPL)

}
