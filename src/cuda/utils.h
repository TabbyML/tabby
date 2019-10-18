#pragma once

#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <NvInfer.h>

namespace ctranslate2 {
  namespace cuda {

#define CUDA_CHECK(ans) { ctranslate2::cuda::cuda_assert((ans), __FILE__, __LINE__); }
#define CUBLAS_CHECK(ans) { ctranslate2::cuda::cublas_assert((ans), __FILE__, __LINE__); }
#define CUDNN_CHECK(ans) { ctranslate2::cuda::cudnn_assert((ans), __FILE__, __LINE__); }

// Default execution policy for Thrust.
#define THRUST_EXECUTION_POLICY thrust::cuda::par(ctranslate2::cuda::get_thrust_allocator()) \
    .on(ctranslate2::cuda::get_cuda_stream())

// Convenience macro to call Thrust functions with the default execution policy.
#define THRUST_CALL(FUN, ...) FUN(THRUST_EXECUTION_POLICY, __VA_ARGS__)

    void cuda_assert(cudaError_t code, const std::string& file, int line);
    void cublas_assert(cublasStatus_t status, const std::string& file, int line);
    void cudnn_assert(cudnnStatus_t status, const std::string& file, int line);

    cudaStream_t get_cuda_stream();
    cublasHandle_t get_cublas_handle();
    cudnnHandle_t get_cudnn_handle();

    int get_gpu_count();
    bool has_gpu();
    bool has_fast_fp16();
    bool has_fast_int8();

    // Custom allocator for Thrust.
    class ThrustAllocator {
    public:
      typedef char value_type;
      value_type* allocate(std::ptrdiff_t num_bytes);
      void deallocate(value_type* p, size_t);
    };

    ThrustAllocator& get_thrust_allocator();

    class TensorRTLayer {
    public:
      virtual ~TensorRTLayer();

    protected:
      void run(void** bindings, const std::vector<nvinfer1::Dims>& input_dims);

      // These methods are called on the first call to run().
      virtual void build_network(nvinfer1::INetworkDefinition* network) = 0;
      virtual void set_optimization_profile(nvinfer1::IOptimizationProfile* profile) = 0;

    private:
      void build();
      nvinfer1::INetworkDefinition* _network = nullptr;
      nvinfer1::ICudaEngine* _engine = nullptr;
      nvinfer1::IExecutionContext* _execution_context = nullptr;
      nvinfer1::IBuilderConfig* _builder_config = nullptr;
    };

    // Statically assiocate cudnnDataType_t with a C++ type.
    template <class T>
    struct TypeToCUDNNType {};

#define MATCH_CUDNN_DATA_TYPE(TYPE, CUDNN_DATA_TYPE)            \
    template<>                                                  \
    struct TypeToCUDNNType<TYPE> {                              \
      static constexpr cudnnDataType_t value = CUDNN_DATA_TYPE; \
    }

    MATCH_CUDNN_DATA_TYPE(float, CUDNN_DATA_FLOAT);
    MATCH_CUDNN_DATA_TYPE(int8_t, CUDNN_DATA_INT8);
    MATCH_CUDNN_DATA_TYPE(int32_t, CUDNN_DATA_INT32);

  }
}
