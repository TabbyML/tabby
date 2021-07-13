#pragma once

#include "ctranslate2/ops/activation.h"
#include "ctranslate2/ops/ops.h"
#include "ctranslate2/models/model.h"

namespace ctranslate2 {
  namespace layers {

    std::pair<StorageView, StorageView>
    make_sequence_inputs(const std::vector<std::vector<size_t>>& ids,
                         const Device device,
                         const dim_t length_multiple_of = 1);

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
      std::unique_ptr<const StorageView> _scale;
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
    protected:
      const StorageView& get_position_encoding(dim_t max_time) override;
    private:
      const StorageView& _encoding;
    };

    // Concrete position encoder computing sinusoidal position encodings (compatible with OpenNMT-tf).
    class SinusoidalPositionEncoder : public PositionEncoder {
    public:
      SinusoidalPositionEncoder(dim_t depth,
                                DataType dtype = DataType::FLOAT,
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
      void mask_weights(const StorageView& index);
      void reset_mask();
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
      const ops::LayerNorm _norm_op;
      const StorageView& _beta;
      const StorageView& _gamma;
    };

  }
}
