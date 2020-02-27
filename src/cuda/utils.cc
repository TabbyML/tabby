#include "./utils.h"

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>

#include "ctranslate2/primitives/primitives.h"
#include "ctranslate2/utils.h"

namespace ctranslate2 {
  namespace cuda {

    std::string cublasGetStatusString(cublasStatus_t status)
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

    cudaStream_t get_cuda_stream() {
      // Only use the default stream for now.
      return static_cast<cudaStream_t>(0);
    }

    class CublasHandle {
    public:
      CublasHandle() {
        CUBLAS_CHECK(cublasCreate(&_handle));
        CUBLAS_CHECK(cublasSetStream(_handle, get_cuda_stream()));
      }
      ~CublasHandle() {
        cublasDestroy(_handle);
      }
      cublasHandle_t get() const {
        return _handle;
      }
    private:
      cublasHandle_t _handle;
    };

    class CudnnHandle {
    public:
      CudnnHandle() {
        CUDNN_CHECK(cudnnCreate(&_handle));
        CUDNN_CHECK(cudnnSetStream(_handle, get_cuda_stream()));
      }
      ~CudnnHandle() {
        cudnnDestroy(_handle);
      }
      cudnnHandle_t get() const {
        return _handle;
      }
    private:
      cudnnHandle_t _handle;
    };

    // We create one cuBLAS/cuDNN handle per host thread. The handle is destroyed
    // when the thread exits.

    cublasHandle_t get_cublas_handle() {
      static thread_local CublasHandle cublas_handle;
      return cublas_handle.get();
    }

    cudnnHandle_t get_cudnn_handle() {
      static thread_local CudnnHandle cudnn_handle;
      return cudnn_handle.get();
    }

    CachingAllocatorConfig get_caching_allocator_config() {
      CachingAllocatorConfig config;
      const char* config_env = std::getenv("CT2_CUDA_CACHING_ALLOCATOR_CONFIG");
      if (config_env) {
        const std::vector<std::string> values = split_string(config_env, ',');
        if (values.size() != 4)
          throw std::invalid_argument("CT2_CUDA_CACHING_ALLOCATOR_CONFIG environment variable "
                                      "should have format: "
                                      "bin_growth,min_bin,max_bin,max_cached_bytes");
        config.bin_growth = std::stoul(values[0]);
        config.min_bin = std::stoul(values[1]);
        config.max_bin = std::stoul(values[2]);
        config.max_cached_bytes = std::stoull(values[3]);
      }
      return config;
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

    static class Logger : public nvinfer1::ILogger {
      void log(Severity severity, const char* msg) override {
        if (static_cast<int>(severity) < static_cast<int>(Severity::kINFO))
          std::cerr << msg << std::endl;
      }
    } g_logger;

    static class Allocator : public nvinfer1::IGpuAllocator {
      void* allocate(uint64_t size, uint64_t, uint32_t) override {
        return primitives<Device::CUDA>::alloc_data(size);
      }

      void free(void* memory) override {
        primitives<Device::CUDA>::free_data(memory);
      }

    } g_allocator;


    bool has_fast_int8() {
      static bool has_fast_int8 = false;
      static std::once_flag flag;
      std::call_once(flag,
                     []() {
                       auto builder = nvinfer1::createInferBuilder(g_logger);
                       has_fast_int8 = builder->platformHasFastInt8();
                       builder->destroy();
                     });
      return has_fast_int8;
    }

    TensorRTLayer::~TensorRTLayer() {
      if (_execution_context) {
        _execution_context->destroy();
        _engine->destroy();
      }
    }

    void TensorRTLayer::build() {
      auto builder = nvinfer1::createInferBuilder(g_logger);
      builder->setGpuAllocator(&g_allocator);
      auto network = builder->createNetworkV2(
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
      build_network(network);
      auto profile = builder->createOptimizationProfile();
      set_optimization_profile(profile);
      auto builder_config = builder->createBuilderConfig();
      builder_config->setMaxWorkspaceSize(1 << 30);
      builder_config->addOptimizationProfile(profile);
      _engine = builder->buildEngineWithConfig(*network, *builder_config);
      _execution_context = _engine->createExecutionContext();
      network->destroy();
      builder_config->destroy();
      builder->destroy();
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
