#include "ctranslate2/layers/common.h"

#include <cmath>

#include "../device_dispatch.h"

namespace ctranslate2 {
  namespace layers {

    Embeddings::Embeddings(const models::Model& model, const std::string& scope)
      : _embeddings(model.get_variable(scope + "/weight"))
      , _qscale(model.get_variable_if_exists(scope + "/weight_scale"))
      , _scale(model.get_flag_with_default(scope + "/multiply_by_sqrt_depth", true)
               ? new StorageView(static_cast<float>(sqrt(_embeddings.dim(-1))))
               : nullptr) {
    }

    void Embeddings::operator()(const StorageView& ids,
                                StorageView& output) {
      PROFILE("Embeddings");
      if (_embeddings.dtype() == DataType::DT_INT16 || _embeddings.dtype() == DataType::DT_INT8) {
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


    static bool should_shift_input_to_u8(Device device, DataType dtype) {
      // If the target Gemm implementation prefers the u8s8s32 format, we can shift
      // the input to the u8 domain and add a compensation term.
      return (device == Device::CPU
              && dtype == DataType::DT_INT8
              && primitives<Device::CPU>::prefer_u8s8s32_gemm());
    }

    static StorageView* compute_u8_compensation(const StorageView& weight) {
      // The compensation term for the shifted input only depends on the weight, so
      // we can compute it once.
      const dim_t k = weight.dim(1);
      const dim_t n = weight.dim(0);
      auto* compensation = new StorageView({n}, DataType::DT_INT32);
      primitives<Device::CPU>::compute_u8_compensation(weight.data<int8_t>(),
                                                       /*transpose=*/true,
                                                       k, n,
                                                       /*alpha=*/1,
                                                       compensation->data<int32_t>());
      return compensation;
    }

    Dense::Dense(const models::Model& model, const std::string& scope)
      : _weight(model.get_variable(scope + "/weight"))
      , _bias(model.get_variable_if_exists(scope + "/bias"))
      , _qscale(model.get_variable_if_exists(scope + "/weight_scale"))
      , _partial_weight(_weight.device(), _weight.dtype())
      , _partial_bias(_weight.device(), DataType::DT_FLOAT)
      , _partial_qscale(_weight.device())
      , _gemm_op(1, 0, false, true)
      , _u8_quantization_shift(should_shift_input_to_u8(_weight.device(), _weight.dtype())
                               ? 128 : 0)
      , _u8_shift_compensation(_u8_quantization_shift != 0
                               ? compute_u8_compensation(_weight) : nullptr) {
    }

    void Dense::mask_weights(const StorageView& index) {
      ops::Gather()(_weight, index, _partial_weight);
      if (_u8_shift_compensation)
        _u8_shift_compensation.reset(compute_u8_compensation(_partial_weight));
      if (_bias)
        ops::Gather()(*_bias, index, _partial_bias);
      if (_qscale && !_qscale->is_scalar())
        ops::Gather()(*_qscale, index, _partial_qscale);
    }

    void Dense::reset_mask() {
      _partial_weight.clear();
      _partial_bias.clear();
      _partial_qscale.clear();
    }

    void Dense::operator()(const StorageView& input, StorageView& output) {
      PROFILE("Dense");
      const StorageView* qscale = _partial_qscale.empty() ? _qscale : &_partial_qscale;
      const StorageView* weight = _partial_weight.empty() ? &_weight : &_partial_weight;
      const StorageView* bias = _partial_bias.empty() ? _bias : &_partial_bias;

      if (_weight.dtype() == DataType::DT_INT16 || _weight.dtype() == DataType::DT_INT8) {
        const auto device = input.device();
        StorageView qinput(_weight.dtype(), device);
        StorageView qinput_scale(_qscale->dtype(), device);
        StorageView qoutput(DataType::DT_INT32, device);
        ops::Quantize()(input, qinput, qinput_scale, _u8_quantization_shift);
        _gemm_op(qinput, *weight, qoutput, _u8_shift_compensation.get());
        ops::Dequantize()(qoutput, qinput_scale, *qscale, output);
      } else {
        _gemm_op(input, *weight, output);
      }

      if (bias) {
        DEVICE_DISPATCH(output.device(),
                        primitives<D>::add_batch_broadcast(bias->data<float>(),
                                                           output.data<float>(),
                                                           bias->size(),
                                                           output.size()));
      }
    }


    LayerNorm::LayerNorm(const models::Model& model, const std::string& scope)
      : _beta(model.get_variable(scope + "/beta"))
      , _gamma(model.get_variable(scope + "/gamma")) {
    }

    void LayerNorm::operator()(const StorageView& input, StorageView& output) {
      _norm_op(_beta, _gamma, input, output);
    }

  }
}
