#include "ctranslate2/layers/common.h"

#include <cmath>

#include "ctranslate2/ops/activation.h"
#include "cpu/backend.h"
#include "dispatch.h"

namespace ctranslate2 {
  namespace layers {

    StorageView
    make_sequence_inputs(const std::vector<std::vector<size_t>>& ids,
                         const Device device,
                         const dim_t length_multiple_of,
                         StorageView* lengths) {
      const dim_t batch_size = ids.size();

      if (lengths)
        *lengths = StorageView({batch_size}, DataType::INT32);

      // Record lengths and maximum length.
      dim_t max_length = 0;
      for (dim_t i = 0; i < batch_size; ++i) {
        const dim_t length = ids[i].size();
        if (lengths)
          lengths->at<int32_t>(i) = length;
        max_length = std::max(max_length, length);
      }

      if (max_length % length_multiple_of != 0) {
        max_length += (length_multiple_of - max_length % length_multiple_of);
      }

      // Make 2D input.
      StorageView input({batch_size, max_length}, int32_t(0));
      for (dim_t i = 0; i < batch_size; ++i) {
        const dim_t length = ids[i].size();
        for (dim_t t = 0; t < length; ++t)
          input.at<int32_t>({i, t}) = ids[i][t];
      }

      if (lengths)
        *lengths = lengths->to(device);
      return input.to(device);
    }


    Embeddings::Embeddings(const models::Model& model, const std::string& scope)
      : _embeddings(model.get_variable(scope + "/weight"))
      , _output_type(get_default_float_type(model.effective_compute_type()))
      , _qscale(model.get_variable_if_exists(scope + "/weight_scale"))
    {
    }

    DataType Embeddings::output_type() const {
      return _output_type;
    }

    dim_t Embeddings::output_size() const {
      return _embeddings.dim(1);
    }

    void Embeddings::operator()(const StorageView& ids,
                                StorageView& output) const {
      PROFILE("Embeddings");
      if (_embeddings.dtype() == DataType::INT16 || _embeddings.dtype() == DataType::INT8) {
        const auto device = output.device();
        StorageView gathered(_embeddings.dtype(), device);
        _gather_op(_embeddings, ids, gathered);
        if (_qscale->is_scalar())
          ops::Dequantize()(gathered, *_qscale, output);
        else {
          StorageView scale(_qscale->dtype(), device);
          _gather_op(*_qscale, ids, scale);
          ops::Dequantize()(gathered, scale, output);
        }
      } else {
        _gather_op(_embeddings, ids, output);
      }
    }


    ParallelEmbeddings::ParallelEmbeddings(const models::Model& model,
                                           const std::string& scope,
                                           const EmbeddingsMerge merge)
      : _merge(merge)
    {
      auto single_layer = build_optional_layer<Embeddings>(model, scope);
      if (single_layer)
        _layers.emplace_back(std::move(single_layer));
      else
        _layers = build_layers_list<const Embeddings>(model, scope);
    }

    DataType ParallelEmbeddings::output_type() const {
      return _layers[0]->output_type();
    }

    dim_t ParallelEmbeddings::output_size() const {
      dim_t size = 0;

      switch (_merge) {
      case EmbeddingsMerge::Concat:
        for (const auto& layer : _layers)
          size += layer->output_size();
        break;
      case EmbeddingsMerge::Add:
        size = _layers[0]->output_size();
        break;
      };

      return size;
    }

    void ParallelEmbeddings::operator()(const std::vector<StorageView>& ids,
                                        StorageView& output) const {
      if (ids.size() != _layers.size())
        throw std::invalid_argument("Expected "
                                    + std::to_string(_layers.size())
                                    + " input features (including the main tokens), but got "
                                    + std::to_string(ids.size())
                                    + " input features instead");

      for (size_t i = 0; i < _layers.size(); ++i) {
        StorageView intermediate(output.device(), output.dtype());
        (*_layers[i])(ids[i], intermediate);

        if (i == 0) {
          output = std::move(intermediate);
        } else {

          switch (_merge) {
          case EmbeddingsMerge::Add: {
            ops::Add()(intermediate, output, output);
            break;
          }

          case EmbeddingsMerge::Concat: {
            StorageView tmp = std::move(output);
            ops::Concat(-1)({&tmp, &intermediate}, output);
            break;
          }
          }

        }
      }
    }


    void PositionEncoder::operator()(StorageView& input, dim_t index) {
      const dim_t time = input.dim(1);
      const dim_t depth = input.dim(-1);
      const dim_t max_time = std::max(time, index + 1);
      const StorageView& encodings = get_position_encoding(max_time);
      const dim_t num_encodings = encodings.dim(0);

      if (max_time > num_encodings)
        throw std::runtime_error("No position encodings are defined for positions >= "
                                 + std::to_string(num_encodings)
                                 + ", but got position "
                                 + std::to_string(max_time - 1));
      if (depth != encodings.dim(1))
        throw std::invalid_argument("Shape mismatch: position encodings have depth "
                                    + std::to_string(encodings.dim(1))
                                    + ", but the input has depth "
                                    + std::to_string(depth));

      DEVICE_AND_TYPE_DISPATCH(input.device(), input.dtype(),
                               primitives<D>::add_batch_broadcast(encodings.data<T>() + index * depth,
                                                                  input.data<T>(),
                                                                  time * depth,
                                                                  input.size()));
    }

    void PositionEncoder::operator()(const StorageView& input, StorageView& output, dim_t index) {
      output = input;
      operator()(output, index);
    }


    PositionEmbedding::PositionEmbedding(const models::Model& model, const std::string& scope)
      : _encoding(model.get_variable(scope + "/encodings"))
    {
    }

    const StorageView& PositionEmbedding::get_position_encoding(dim_t) {
      return _encoding;
    }

    DataType PositionEmbedding::output_type() const {
      return _encoding.dtype();
    }

    dim_t PositionEmbedding::output_size() const {
      return _encoding.dim(1);
    }

    dim_t PositionEmbedding::num_positions() const {
      return _encoding.dim(0);
    }


    static StorageView generate_sinusoidal_position_encoding(dim_t max_time,
                                                             dim_t depth,
                                                             DataType dtype,
                                                             Device device) {
      const float log_timescale_increment = std::log(10000.f) / static_cast<float>(depth / 2 - 1);
      StorageView timescales({depth / 2}, -log_timescale_increment);
      for (dim_t i = 0; i < timescales.size(); ++i)
        timescales.at<float>(i) = std::exp(timescales.at<float>(i) * i);

      StorageView scaled_time({max_time, depth / 2});
      for (dim_t i = 0; i < scaled_time.dim(0); ++i) {
        for (dim_t j = 0; j < scaled_time.dim(1); ++j) {
          scaled_time.at<float>({i, j}) = (i + 1) * timescales.at<float>(j);
        }
      }

      StorageView sin_encoding;
      StorageView cos_encoding;

      ops::Sin()(scaled_time, sin_encoding);
      ops::Cos()(scaled_time, cos_encoding);

      StorageView encoding;
      ops::Concat(-1)({&sin_encoding, &cos_encoding}, encoding);
      return encoding.to(dtype).to(device);
    }

    SinusoidalPositionEncoder::SinusoidalPositionEncoder(dim_t depth, DataType dtype, Device device)
      : _encoding(generate_sinusoidal_position_encoding(500, depth, dtype, device))
    {
    }

    const StorageView& SinusoidalPositionEncoder::get_position_encoding(dim_t max_time) {
      if (max_time > _encoding.dim(0))
        _encoding = generate_sinusoidal_position_encoding(max_time,
                                                          _encoding.dim(1),
                                                          _encoding.dtype(),
                                                          _encoding.device());
      return _encoding;
    }

    DataType SinusoidalPositionEncoder::output_type() const {
      return _encoding.dtype();
    }

    dim_t SinusoidalPositionEncoder::output_size() const {
      return _encoding.dim(1);
    }


    static const StorageView& get_linear_weight(const models::Model& model,
                                                const std::string& scope,
                                                bool* is_packed) {
      const StorageView* weight = model.get_variable_if_exists(scope + "/weight_packed");
      if (weight) {
        *is_packed = true;
        return *weight;
      }
      *is_packed = false;
      return model.get_variable(scope + "/weight");
    }

    Dense::Dense(const models::Model& model,
                 const std::string& scope,
                 const ops::ActivationType* activation_type)
      : _packed_weight(false)
      , _weight(get_linear_weight(model, scope, &_packed_weight))
      , _bias(model.get_variable_if_exists(scope + "/bias"))
      , _qscale(model.get_variable_if_exists(scope + "/weight_scale"))
      , _u8_shift_compensation((_weight.device() == Device::CPU
                                && _weight.dtype() == DataType::INT8
                                && cpu::prefer_u8s8s32_gemm())
                               ? &model.get_variable(scope + "/weight_compensation")
                               : nullptr)
      , _partial_weight(_weight.device(), _weight.dtype())
      , _partial_bias(_weight.device(), _bias ? _bias->dtype() : DataType::FLOAT32)
      , _partial_qscale(_weight.device(), DataType::FLOAT32)
      , _partial_u8_shift_compensation(_weight.device(), DataType::INT32)
      , _output_type(get_default_float_type(model.effective_compute_type()))
      , _quantized_gemm(_weight.dtype() == DataType::INT16 || _weight.dtype() == DataType::INT8)
      , _gemm_op(/*alpha=*/1,
                 /*beta=*/0,
                 /*trans_a=*/false,
                 /*trans_b=*/true,
                 /*a_is_packed=*/false,
                 _packed_weight,
                 _quantized_gemm ? nullptr : activation_type)
      , _quantize_op(model.use_global_int16_scale()
                     ? ops::Quantize::ScaleType::GLOBAL
                     : ops::Quantize::ScaleType::PER_LAYER,
                     /*shift_to_uint8=*/bool(_u8_shift_compensation),
                     /*round_before_cast=*/model.round_before_cast_in_quantization())
      , _dequantize_op(activation_type)
    {
    }

    DataType Dense::output_type() const {
      return _output_type;
    }

    dim_t Dense::output_size() const {
      return _partial_weight ? _partial_weight.dim(0) : _weight.dim(0);
    }

    void Dense::select_weights(const StorageView* index, const StorageView* extra_bias) {
      if (index) {
        if (_packed_weight)
          throw std::runtime_error("Can't select pre-packed weight");
        ops::Gather()(_weight, *index, _partial_weight);

        if (_bias) {
          ops::Gather()(*_bias, *index, _partial_bias);
          if (extra_bias)
            ops::Add()(_partial_bias, *extra_bias, _partial_bias);
        } else if (extra_bias) {
          _partial_bias = *extra_bias;
        }

        if (_u8_shift_compensation)
          ops::Gather()(*_u8_shift_compensation, *index, _partial_u8_shift_compensation);
        if (_qscale && !_qscale->is_scalar())
          ops::Gather()(*_qscale, *index, _partial_qscale);
      } else {
        _partial_weight.clear();
        _partial_bias.clear();
        _partial_qscale.clear();
        _partial_u8_shift_compensation.clear();
      }
    }

    void Dense::operator()(const StorageView& input, StorageView& output) const {
      PROFILE("Dense");
      const StorageView* qscale = _partial_qscale.empty() ? _qscale : &_partial_qscale;
      const StorageView* weight = _partial_weight.empty() ? &_weight : &_partial_weight;
      const StorageView* bias = _partial_bias.empty() ? _bias : &_partial_bias;
      const StorageView* compensation = (_partial_u8_shift_compensation.empty()
                                         ? _u8_shift_compensation
                                         : &_partial_u8_shift_compensation);
      if (_quantized_gemm) {
        const auto device = input.device();
        StorageView qinput(_weight.dtype(), device);
        StorageView qinput_scale(_qscale->dtype(), device);
        StorageView qoutput(DataType::INT32, device);
        _quantize_op(input, qinput, qinput_scale);
        _gemm_op(qinput, *weight, qoutput, compensation);
        _dequantize_op(qoutput,
                       qinput_scale,
                       *qscale,
                       /*trans_a=*/false,
                       /*trans_b=*/true,
                       output,
                       bias);
      } else {
        _gemm_op(input, *weight, output, nullptr, bias);
      }
    }


    LayerNorm::LayerNorm(const models::Model& model, const std::string& scope)
      : _beta(model.get_variable_if_exists(scope + "/beta"))
      , _gamma(model.get_variable(scope + "/gamma")) {
    }

    DataType LayerNorm::output_type() const {
      return _gamma.dtype();
    }

    dim_t LayerNorm::output_size() const {
      return _gamma.size();
    }

    void LayerNorm::operator()(const StorageView& input, StorageView& output) const {
      if (_beta)
        ops::LayerNorm()(*_beta, _gamma, input, output);
      else
        ops::RMSNorm()(_gamma, input, output);
    }


    Conv1D::Conv1D(const models::Model& model,
                   const std::string& scope,
                   dim_t stride,
                   dim_t padding,
                   dim_t dilation)
      : _conv_op(stride, padding, dilation)
      , _weight(model.get_variable(scope + "/weight"))
      , _bias(model.get_variable_if_exists(scope + "/bias")) {
    }

    DataType Conv1D::output_type() const {
      return _weight.dtype();
    }

    dim_t Conv1D::output_size() const {
      return _weight.dim(0);
    }

    dim_t Conv1D::input_size() const {
      return _weight.dim(1);
    }

    void Conv1D::operator()(const StorageView& input, StorageView& output) const {
      if (_bias)
        _conv_op(input, _weight, *_bias, output);
      else
        _conv_op(input, _weight, output);
    }

  }
}
