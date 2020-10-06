#include "ctranslate2/layers/common.h"

#include <cmath>

#include "device_dispatch.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace layers {

    std::pair<StorageView, StorageView>
    make_sequence_inputs(const std::vector<std::vector<size_t>>& ids,
                         const Device device,
                         const dim_t length_multiple_of) {
      const dim_t batch_size = ids.size();

      // Record lengths and maximum length.
      dim_t max_length = 0;
      StorageView lengths({batch_size}, DataType::INT32);
      for (dim_t i = 0; i < batch_size; ++i) {
        const dim_t length = ids[i].size();
        lengths.at<int32_t>(i) = length;
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

      return std::make_pair(input.to(device), lengths.to(device));
    }

    static StorageView* get_sqrt_depth_scale(const StorageView& embeddings) {
      const auto scale = std::sqrt(static_cast<float>(embeddings.dim(-1)));
      if (embeddings.dtype() == DataType::FLOAT16) {
        return new StorageView(float16_t(scale));
      } else {
        return new StorageView(scale);
      }
    }

    Embeddings::Embeddings(const models::Model& model, const std::string& scope)
      : _embeddings(model.get_variable(scope + "/weight"))
      , _qscale(model.get_variable_if_exists(scope + "/weight_scale"))
      , _scale(model.get_flag_with_default(scope + "/multiply_by_sqrt_depth", true)
               ? get_sqrt_depth_scale(_embeddings)
               : nullptr) {
    }

    DataType Embeddings::output_type() const {
      return _embeddings.dtype() == DataType::FLOAT16 ? DataType::FLOAT16 : DataType::FLOAT;
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

      if (_scale)
        ops::Mul()(output, *_scale, output);
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

    Dense::Dense(const models::Model& model, const std::string& scope)
      : _packed_weight(false)
      , _weight(get_linear_weight(model, scope, &_packed_weight))
      , _bias(model.get_variable_if_exists(scope + "/bias"))
      , _qscale(model.get_variable_if_exists(scope + "/weight_scale"))
      , _u8_shift_compensation(model.get_variable_if_exists(scope + "/weight_compensation"))
      , _partial_weight(_weight.device(), _weight.dtype())
      , _partial_bias(_weight.device(), _bias ? _bias->dtype() : DataType::FLOAT)
      , _partial_qscale(_weight.device(), DataType::FLOAT)
      , _partial_u8_shift_compensation(_weight.device(), DataType::INT32)
      , _gemm_op(/*alpha=*/1,
                 /*beta=*/0,
                 /*trans_a=*/false,
                 /*trans_b=*/true,
                 /*a_is_packed=*/false,
                 _packed_weight)
      , _quantize_op(/*int16_scale_type=*/ops::Quantize::ScaleType::GLOBAL,
                     /*shit_to_uint8=*/bool(_u8_shift_compensation)) {
    }

    DataType Dense::output_type() const {
      return _weight.dtype() == DataType::FLOAT16 ? DataType::FLOAT16 : DataType::FLOAT;
    }

    dim_t Dense::output_size() const {
      return _weight.dim(0);
    }

    void Dense::mask_weights(const StorageView& index) {
      if (_packed_weight)
        throw std::runtime_error("Can't mask pre-packed weight");
      ops::Gather()(_weight, index, _partial_weight);
      if (_u8_shift_compensation)
        ops::Gather()(*_u8_shift_compensation, index, _partial_u8_shift_compensation);
      if (_bias)
        ops::Gather()(*_bias, index, _partial_bias);
      if (_qscale && !_qscale->is_scalar())
        ops::Gather()(*_qscale, index, _partial_qscale);
    }

    void Dense::reset_mask() {
      _partial_weight.clear();
      _partial_bias.clear();
      _partial_qscale.clear();
      _partial_u8_shift_compensation.clear();
    }

    void Dense::operator()(const StorageView& input, StorageView& output) const {
      PROFILE("Dense");
      const StorageView* qscale = _partial_qscale.empty() ? _qscale : &_partial_qscale;
      const StorageView* weight = _partial_weight.empty() ? &_weight : &_partial_weight;
      const StorageView* bias = _partial_bias.empty() ? _bias : &_partial_bias;
      const StorageView* compensation = (_partial_u8_shift_compensation.empty()
                                         ? _u8_shift_compensation
                                         : &_partial_u8_shift_compensation);

      if (_weight.dtype() == DataType::INT16 || _weight.dtype() == DataType::INT8) {
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
                       output);
      } else {
        _gemm_op(input, *weight, output);
      }

      if (bias) {
        DEVICE_DISPATCH(output.device(),
                        TYPE_DISPATCH(bias->dtype(),
                                      primitives<D>::add_batch_broadcast(bias->data<T>(),
                                                                         output.data<T>(),
                                                                         bias->size(),
                                                                         output.size())));
      }
    }


    LayerNorm::LayerNorm(const models::Model& model, const std::string& scope)
      : _beta(model.get_variable(scope + "/beta"))
      , _gamma(model.get_variable(scope + "/gamma")) {
    }

    DataType LayerNorm::output_type() const {
      return _beta.dtype();
    }

    dim_t LayerNorm::output_size() const {
      return _beta.size();
    }

    void LayerNorm::operator()(const StorageView& input, StorageView& output) const {
      _norm_op(_beta, _gamma, input, output);
    }

  }
}
