#include "./utils.h"

#include <iostream>
#include <stdexcept>

#include "ctranslate2/primitives/primitives.h"

namespace ctranslate2 {
  namespace cuda {

    static std::string cublasGetStatusString(cublasStatus_t status)
    {
      switch (status)
      {
      case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
      case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
      case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
      case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
      case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
      case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
      case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
      case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
      case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
      case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
      default:
        return "UNKNOWN";
      }
    }

    void cuda_assert(cudaError_t code, const std::string& file, int line)
    {
      if (code != cudaSuccess)
        throw std::runtime_error("CUDA failed with error '"
                                 + std::string(cudaGetErrorString(code))
                                 + "' at " + file + ":" + std::to_string(line));
    }

    void cublas_assert(cublasStatus_t status, const std::string& file, int line)
    {
      if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS failed with status '"
                                 + cublasGetStatusString(status)
                                 + "' at " + file + ":" + std::to_string(line));
    }

    void cudnn_assert(cudnnStatus_t code, const std::string& file, int line)
    {
      if (code != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error("cuDNN failed with error '"
                                 + std::string(cudnnGetErrorString(code))
                                 + "' at " + file + ":" + std::to_string(line));
    }

    cudaStream_t get_cuda_stream() {
      // Use one CUDA stream per host thread.
      static thread_local cudaStream_t stream;
      static thread_local bool initialized = false;
      if (!initialized) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        initialized = true;
      }
      return stream;
    }

    cublasHandle_t get_cublas_handle() {
      // Use one cuBLAS handle per host thread.
      static thread_local cublasHandle_t handle;
      static thread_local bool initialized = false;
      if (!initialized) {
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetStream(handle, get_cuda_stream()));
        initialized = true;
      }
      return handle;
    }

    cudnnHandle_t get_cudnn_handle() {
      // Use one cuDNN handle per host thread.
      static thread_local cudnnHandle_t handle;
      static thread_local bool initialized = false;
      if (!initialized) {
        CUDNN_CHECK(cudnnCreate(&handle));
        CUDNN_CHECK(cudnnSetStream(handle, get_cuda_stream()));
        initialized = true;
      }
      return handle;
    }

    int get_gpu_count() {
      int gpu_count = 0;
      cudaError_t status = cudaGetDeviceCount(&gpu_count);
      if (status != cudaSuccess)
        return 0;
      return gpu_count;
    }

    bool has_gpu() {
      return get_gpu_count() > 0;
    }

    ThrustAllocator::value_type* ThrustAllocator::allocate(std::ptrdiff_t num_bytes) {
      return reinterpret_cast<ThrustAllocator::value_type*>(
        primitives<Device::CUDA>::alloc_data(num_bytes));
    }

    void ThrustAllocator::deallocate(ThrustAllocator::value_type* p, size_t) {
      return primitives<Device::CUDA>::free_data(p);
    }

    ThrustAllocator& get_thrust_allocator() {
      static ThrustAllocator thrust_allocator;
      return thrust_allocator;
    }

    class Logger : public nvinfer1::ILogger {
      void log(Severity severity, const char* msg) override {
        if (static_cast<int>(severity) < static_cast<int>(Severity::kINFO))
          std::cerr << msg << std::endl;
      }
    } g_logger;

    class Allocator : public nvinfer1::IGpuAllocator {
      void* allocate(uint64_t size, uint64_t, uint32_t) override {
        return primitives<Device::CUDA>::alloc_data(size);
      }

      void free(void* memory) override {
        primitives<Device::CUDA>::free_data(memory);
      }

    } g_allocator;

    static nvinfer1::IBuilder* get_trt_builder() {
      static thread_local nvinfer1::IBuilder* builder = nullptr;
      if (!builder) {
        builder = nvinfer1::createInferBuilder(g_logger);
        builder->setGpuAllocator(&g_allocator);
      }
      return builder;
    }

    bool has_fast_fp16() {
      return get_trt_builder()->platformHasFastFp16();
    }

    bool has_fast_int8() {
      return get_trt_builder()->platformHasFastInt8();
    }

    TensorRTLayer::~TensorRTLayer() {
      if (_execution_context) {
        _execution_context->destroy();
        _network->destroy();
        _engine->destroy();
        _builder_config->destroy();
      }
    }

    void TensorRTLayer::build() {
      auto builder = get_trt_builder();
      _network = builder->createNetworkV2(
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
      build_network(_network);
      auto profile = builder->createOptimizationProfile();
      set_optimization_profile(profile);
      _builder_config = builder->createBuilderConfig();
      _builder_config->setMaxWorkspaceSize(1 << 30);
      _builder_config->addOptimizationProfile(profile);
      _engine = builder->buildEngineWithConfig(*_network, *_builder_config);
      _execution_context = _engine->createExecutionContext();
    }

    void TensorRTLayer::run(void** bindings, const std::vector<nvinfer1::Dims>& input_dims) {
      if (!_execution_context)
        build();
      for (size_t i = 0; i < input_dims.size(); ++i)
        _execution_context->setBindingDimensions(i, input_dims[i]);
      _execution_context->enqueueV2(bindings, get_cuda_stream(), nullptr);
    }

  }
}
