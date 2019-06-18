#pragma once

#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <NvInfer.h>

namespace ctranslate2 {
  namespace cuda {

#define CUDA_CHECK(ans) { ctranslate2::cuda::cuda_assert((ans), __FILE__, __LINE__); }
#define CUBLAS_CHECK(ans) { ctranslate2::cuda::cublas_assert((ans), __FILE__, __LINE__); }
#define CUDNN_CHECK(ans) { ctranslate2::cuda::cudnn_assert((ans), __FILE__, __LINE__); }

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

    class TensorRTLayer {
    public:
      virtual ~TensorRTLayer();

    protected:
      virtual void build_network(nvinfer1::INetworkDefinition* network) = 0;
      void run(int batch_size, void** bindings);
      void build(bool force = false);
      void clear();

    private:
      nvinfer1::INetworkDefinition* _network = nullptr;
      nvinfer1::ICudaEngine* _engine = nullptr;
      nvinfer1::IExecutionContext* _execution_context = nullptr;
    };

  }
}
