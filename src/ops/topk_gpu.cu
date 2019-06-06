#include "ctranslate2/ops/topk.h"

#include <iostream>
#include <memory>

#include <NvInfer.h>

#include "ctranslate2/cuda/utils.h"

namespace ctranslate2 {
  namespace ops {

    class Logger : public nvinfer1::ILogger {
      void log(Severity severity, const char* msg) override {
        // suppress info-level messages
        if (severity != Severity::kINFO)
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

    class TopKLayer {
    public:
      TopKLayer(nvinfer1::IBuilder* builder, int k, int depth)
        : _depth(depth) {
        _network = builder->createNetwork();
        nvinfer1::Dims input_dim{1, {depth}, {nvinfer1::DimensionType::kCHANNEL}};
        nvinfer1::ITensor* input = _network->addInput("x", nvinfer1::DataType::kFLOAT, input_dim);
        nvinfer1::ITopKLayer* topk = _network->addTopK(*input, nvinfer1::TopKOperation::kMAX, k, 1);
        nvinfer1::ITensor* values_t = topk->getOutput(0);
        nvinfer1::ITensor* indices_t = topk->getOutput(1);
        _network->markOutput(*values_t);
        _network->markOutput(*indices_t);
        values_t->setName("values");
        indices_t->setName("indices");
        indices_t->setType(nvinfer1::DataType::kINT32);
        _engine = builder->buildCudaEngine(*_network);
        _execution_context = _engine->createExecutionContext();
      }

      ~TopKLayer() {
        _network->destroy();
        _engine->destroy();
      }

      nvinfer1::IExecutionContext* get_execution_context() {
        return _execution_context;
      }

      int depth() const {
        return _depth;
      }

    private:
      int _depth;
      nvinfer1::INetworkDefinition* _network;
      nvinfer1::ICudaEngine* _engine;
      nvinfer1::IExecutionContext* _execution_context;
    };

    static int max_batch_size = 512;

    template <Device D, typename DataType, typename IndexType>
    void TopK::compute(const StorageView& x,
                       StorageView& values,
                       StorageView& indices) const {
      static thread_local nvinfer1::IBuilder* builder = nullptr;
      if (!builder) {
        builder = nvinfer1::createInferBuilder(g_logger);
        builder->setMaxBatchSize(max_batch_size);
        builder->setMaxWorkspaceSize(1 << 30);
        builder->setGpuAllocator(&g_allocator);
      }

      int depth = x.dim(-1);
      int batch_size = x.size() / depth;

      if (batch_size > max_batch_size)
        throw std::runtime_error("Maximum batch size supported by the TopK layer is "
                                 + std::to_string(max_batch_size) + ", but got "
                                 + std::to_string(batch_size));

      static thread_local std::unique_ptr<TopKLayer> topk_layer;
      if (!topk_layer || topk_layer->depth() != depth)
        topk_layer.reset(new TopKLayer(builder, _k, depth));

      void* bindings[3] = {
        const_cast<DataType*>(x.data<DataType>()),
        values.data<DataType>(),
        indices.data<IndexType>()
      };

      topk_layer->get_execution_context()->enqueue(batch_size,
                                                   bindings,
                                                   cuda::get_cuda_stream(),
                                                   NULL);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    TopK::compute<Device::CUDA, T, int32_t>(const StorageView& x,       \
                                            StorageView& values,        \
                                            StorageView& indices) const;

    DECLARE_IMPL(float)

  }
}
