#pragma once

#include "ctranslate2/ops/activation.h"
#include "ctranslate2/ops/ops.h"
#include "ctranslate2/models/model.h"

namespace ctranslate2 {
  namespace layers {

    StorageView
    make_sequence_inputs(const std::vector<std::vector<size_t>>& ids,
                         const Device device,
                         const dim_t length_multiple_of = 1,
                         StorageView* lengths = nullptr);

    template <typename Layer, typename... Args>
    std::unique_ptr<Layer> build_optional_layer(const models::Model& model,
                                                const std::string& scope,
                                                Args&&... args) {
      if (!model.layer_exists(scope))
        return nullptr;
      return std::make_unique<Layer>(model, scope, std::forward<Args>(args)...);
    }

    template <typename Layer, typename... Args>
    std::vector<std::unique_ptr<Layer>> build_layers_list(const models::Model& model,
                                                          const std::string& prefix,
                                                          Args... args) {
      std::vector<std::unique_ptr<Layer>> layers;

      for (size_t i = 0;; ++i) {
        const std::string layer_scope = prefix + "_" + std::to_string(i);
        if (i == 0)
          layers.emplace_back(std::make_unique<Layer>(model, layer_scope, args...));
        else {
          auto layer = build_optional_layer<Layer>(model, layer_scope, args...);
          if (!layer)
            break;
          layers.emplace_back(std::move(layer));
        }
      }

      return layers;
    }

    class Layer {
    public:
      virtual ~Layer() = default;
      virtual DataType output_type() const = 0;
      virtual dim_t output_size() const = 0;
    };

    class Embeddings : public Layer
    {
    public:
      Embeddings(const models::Model& model, const std::string& scope);
      DataType output_type() const override;
      dim_t output_size() const override;
      void operator()(const StorageView& ids, StorageView& output) const;
    private:
      const ops::Gather _gather_op;
      const StorageView& _embeddings;
      const DataType _output_type;
      const StorageView* _qscale;
    };

    // This enum order should remain fixed.
    enum class EmbeddingsMerge {
      Concat,
      Add,
    };

    class ParallelEmbeddings : public Layer {
    public:
      ParallelEmbeddings(const models::Model& model,
                         const std::string& scope,
                         const EmbeddingsMerge merge);
      size_t num_inputs() const {
        return _layers.size();
      }
      DataType output_type() const override;
      dim_t output_size() const override;
      void operator()(const std::vector<StorageView>& ids, StorageView& output) const;
    private:
      const EmbeddingsMerge _merge;
      std::vector<std::unique_ptr<const Embeddings>> _layers;
    };

    // Base class for position encoders.
    class PositionEncoder : public Layer {
    public:
      void operator()(StorageView& input, dim_t index = 0);
      void operator()(const StorageView& input, StorageView& output, dim_t index = 0);
    protected:
      virtual const StorageView& get_position_encoding(dim_t max_time) = 0;
    };

    // Concrete position encoder loading encoding vectors from the model.
    class PositionEmbedding : public PositionEncoder {
    public:
      PositionEmbedding(const models::Model& model, const std::string& scope);
      DataType output_type() const override;
      dim_t output_size() const override;
      dim_t num_positions() const;
    protected:
      const StorageView& get_position_encoding(dim_t max_time) override;
    private:
      const StorageView& _encoding;
    };

    // Concrete position encoder computing sinusoidal position encodings (compatible with OpenNMT-tf).
    class SinusoidalPositionEncoder : public PositionEncoder {
    public:
      SinusoidalPositionEncoder(dim_t depth,
                                DataType dtype = DataType::FLOAT32,
                                Device device = Device::CPU);
      DataType output_type() const override;
      dim_t output_size() const override;
    protected:
      const StorageView& get_position_encoding(dim_t max_time) override;
    private:
      StorageView _encoding;
    };

    class Dense : public Layer
    {
    public:
      Dense(const models::Model& model,
            const std::string& scope,
            const ops::ActivationType* activation_type = nullptr);
      DataType output_type() const override;
      dim_t output_size() const override;
      void operator()(const StorageView& input, StorageView& output) const;
      void select_weights(const StorageView* index, const StorageView* extra_bias = nullptr);
    private:
      bool _packed_weight;
      const StorageView& _weight;
      const StorageView* _bias;
      const StorageView* _qscale;
      const StorageView* _u8_shift_compensation;
      StorageView _partial_weight;
      StorageView _partial_bias;
      StorageView _partial_qscale;
      StorageView _partial_u8_shift_compensation;
      const DataType _output_type;
      const bool _quantized_gemm;
      const ops::Gemm _gemm_op;
      const ops::Quantize _quantize_op;
      const ops::Dequantize _dequantize_op;
    };

    class LayerNorm : public Layer
    {
    public:
      LayerNorm(const models::Model& model, const std::string& scope);
      DataType output_type() const override;
      dim_t output_size() const override;
      void operator()(const StorageView& input, StorageView& output) const;
    private:
      const StorageView* _beta;
      const StorageView& _gamma;
    };

    class Conv1D : public Layer {
    public:
      Conv1D(const models::Model& model,
             const std::string& scope,
             dim_t stride = 1,
             dim_t padding = 0,
             dim_t dilation = 1);
      DataType output_type() const override;
      dim_t output_size() const override;
      dim_t input_size() const;
      void operator()(const StorageView& input, StorageView& output) const;
    private:
      const ops::Conv1D _conv_op;
      const StorageView& _weight;
      const StorageView* _bias;
    };

  }
}
