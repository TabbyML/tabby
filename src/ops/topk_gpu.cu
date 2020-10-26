#include "ctranslate2/ops/topk.h"

#include <unordered_map>

#include "../cuda/utils.h"

namespace ctranslate2 {
  namespace ops {

#ifdef CT2_WITH_TENSORRT
    template <typename T>
    class TopKLayer : public cuda::TensorRTLayer {
    public:
      TopKLayer(dim_t k)
        : _k(k)
        , _first_depth(0) {
      }

      void operator()(const StorageView& x, StorageView& values, StorageView& indices) {
        const dim_t depth = x.dim(-1);
        const dim_t batch_size = x.size() / depth;

        if (_first_depth == 0)
          _first_depth = depth;

        void* bindings[3] = {
          const_cast<T*>(x.data<T>()),
          values.data<T>(),
          indices.data<int32_t>()
        };

        run(bindings, {nvinfer1::Dims2(batch_size, depth)});
      }

    protected:
      void build_network(nvinfer1::INetworkDefinition* network) override {
        nvinfer1::ITensor* input = network->addInput("x",
                                                     cuda::TensorRTType<T>::type,
                                                     nvinfer1::Dims2(-1, -1));
        nvinfer1::ITopKLayer* topk = network->addTopK(*input, nvinfer1::TopKOperation::kMAX, _k, 2);
        nvinfer1::ITensor* values_t = topk->getOutput(0);
        nvinfer1::ITensor* indices_t = topk->getOutput(1);
        network->markOutput(*values_t);
        network->markOutput(*indices_t);
        values_t->setName("values");
        values_t->setType(cuda::TensorRTType<T>::type);
        indices_t->setName("indices");
        indices_t->setType(nvinfer1::DataType::kINT32);
      }

      void set_builder_config(nvinfer1::IBuilderConfig* config) override {
        config->setMaxWorkspaceSize(1 << 30);
        if (std::is_same<T, float16_t>::value) {
          config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
      }

      void set_optimization_profile(nvinfer1::IOptimizationProfile* profile) override {
        // Optimize for the first seen depth which covers the standard use case
        // of running TopK over a static vocabulary size.
        profile->setDimensions("x",
                               nvinfer1::OptProfileSelector::kMIN,
                               nvinfer1::Dims2(1, _first_depth));
        profile->setDimensions("x",
                               nvinfer1::OptProfileSelector::kOPT,
                               nvinfer1::Dims2(64, _first_depth));
        profile->setDimensions("x",
                               nvinfer1::OptProfileSelector::kMAX,
                               nvinfer1::Dims2(1024, _first_depth));
      }

    private:
      dim_t _k;
      dim_t _first_depth;
    };

    template <typename DataType>
    static TopKLayer<DataType>& get_trt_topk_layer(dim_t k) {
      static thread_local std::unordered_map<dim_t, TopKLayer<DataType>> layer_cache;
      auto it = layer_cache.find(k);
      if (it == layer_cache.end())
        it = layer_cache.emplace(k, TopKLayer<DataType>(k)).first;
      return it->second;
    }
#endif

    template <Device D, typename DataType, typename IndexType>
    void TopK::compute(const StorageView& x,
                       StorageView& values,
                       StorageView& indices) const {
#ifdef CT2_WITH_TENSORRT
      get_trt_topk_layer<DataType>(_k)(x, values, indices);
#else
      if (_k > 1)
        throw std::runtime_error("TopK with k > 1 requires TensorRT");
      const dim_t depth = x.dim(-1);
      const dim_t batch_size = x.size() / depth;
      primitives<D>::row_max(x.data<DataType>(),
                             batch_size,
                             depth,
                             values.data<DataType>(),
                             indices.data<IndexType>());
#endif
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    TopK::compute<Device::CUDA, T, int32_t>(const StorageView& x,       \
                                            StorageView& values,        \
                                            StorageView& indices) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)

  }
}
