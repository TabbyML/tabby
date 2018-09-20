#include "ctranslate2/ops/topk.h"

#include <iostream>

#include <NvInfer.h>

#include "ctranslate2/cuda/utils.h"

using namespace nvinfer1;

namespace ctranslate2 {
  namespace ops {

    class Logger : public ILogger {
      void log(Severity severity, const char* msg) override {
        // suppress info-level messages
        if (severity != Severity::kINFO)
          std::cerr << msg << std::endl;
      }
    } g_logger;

    static int max_batch_size = 512;

    template <Device D, typename DataType, typename IndexType>
    void TopK::compute(const StorageView& x,
                       StorageView& values,
                       StorageView& indices) const {
      int depth = x.dim(-1);
      int batch_size = x.size() / depth;

      if (batch_size > max_batch_size)
        throw std::runtime_error("Maximum batch size supported by the TopK layer is "
                                 + std::to_string(max_batch_size) + ", but got "
                                 + std::to_string(batch_size));

      static thread_local IExecutionContext* context = nullptr;
      if (context == nullptr) {
        IBuilder* builder = createInferBuilder(g_logger);
        builder->setMaxBatchSize(max_batch_size);
        builder->setMaxWorkspaceSize(1 << 30);
        INetworkDefinition* network = builder->createNetwork();
        ITensor* data = network->addInput("x",
                                          nvinfer1::DataType::kFLOAT,
                                          Dims{1, {depth}, {DimensionType::kCHANNEL}});
        ITopKLayer* topk = network->addTopK(*data, TopKOperation::kMAX, _k, 1);
        ITensor* values_t = topk->getOutput(0);
        ITensor* indices_t = topk->getOutput(1);
        network->markOutput(*values_t);
        network->markOutput(*indices_t);
        values_t->setName("values");
        indices_t->setName("indices");
        indices_t->setType(nvinfer1::DataType::kINT32);
        ICudaEngine* engine = builder->buildCudaEngine(*network);
        context = engine->createExecutionContext();
      }

      void* bindings[3] = {
        const_cast<DataType*>(x.data<DataType>()),
        values.data<DataType>(),
        indices.data<IndexType>()
      };

      context->enqueue(batch_size, bindings, cuda::get_cuda_stream(), NULL);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    TopK::compute<Device::CUDA, T, int32_t>(const StorageView& x,       \
                                            StorageView& values,        \
                                            StorageView& indices) const;

    DECLARE_IMPL(float)

  }
}
