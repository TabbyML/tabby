#include "ctranslate2/ops/topk.h"

#include "../cuda/utils.h"

namespace ctranslate2 {
  namespace ops {

    class TopKLayer : public cuda::TensorRTLayer {
    public:
      TopKLayer(int k)
        : _k(k)
        , _depth(0) {
      }

      void operator()(const StorageView& x, StorageView& values, StorageView& indices) {
        int depth = x.dim(-1);
        int batch_size = x.size() / depth;

        if (depth != _depth) {
          _depth = depth;
          build(/*force=*/true);
        }

        void* bindings[3] = {
          const_cast<float*>(x.data<float>()),
          values.data<float>(),
          indices.data<int32_t>()
        };

        run(batch_size, bindings);
      }

    protected:
      void build_network(nvinfer1::INetworkDefinition* network) override {
        nvinfer1::Dims input_dim{1, {_depth}, {nvinfer1::DimensionType::kCHANNEL}};
        nvinfer1::ITensor* input = network->addInput("x", nvinfer1::DataType::kFLOAT, input_dim);
        nvinfer1::ITopKLayer* topk = network->addTopK(*input, nvinfer1::TopKOperation::kMAX, _k, 1);
        nvinfer1::ITensor* values_t = topk->getOutput(0);
        nvinfer1::ITensor* indices_t = topk->getOutput(1);
        network->markOutput(*values_t);
        network->markOutput(*indices_t);
        values_t->setName("values");
        indices_t->setName("indices");
        indices_t->setType(nvinfer1::DataType::kINT32);
      }

    private:
      int _k;
      int _depth;
    };

    template <Device D, typename DataType, typename IndexType>
    void TopK::compute(const StorageView& x,
                       StorageView& values,
                       StorageView& indices) const {
      static thread_local TopKLayer topk_layer(_k);
      topk_layer(x, values, indices);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    TopK::compute<Device::CUDA, T, int32_t>(const StorageView& x,       \
                                            StorageView& values,        \
                                            StorageView& indices) const;

    DECLARE_IMPL(float)

  }
}
