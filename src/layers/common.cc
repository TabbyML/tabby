#include "ctranslate2/layers/common.h"

#include "device_dispatch.h"

namespace ctranslate2 {
  namespace layers {

    Embeddings::Embeddings(const models::Model& model, const std::string& scope)
      : _embeddings(model.get_variable(scope + "/weight"))
      , _qscale(model.get_variable_if_exists(scope + "/weight_scale")) {
    }

    void Embeddings::operator()(const StorageView& ids,
                                StorageView& output) {
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
    }


    Dense::Dense(const models::Model& model, const std::string& scope)
      : _weight(model.get_variable(scope + "/weight"))
      , _bias(model.get_variable_if_exists(scope + "/bias"))
      , _qscale(model.get_variable_if_exists(scope + "/weight_scale"))
      , _partial_weight(_weight.device(), _weight.dtype())
      , _partial_bias(_weight.device(), DataType::DT_FLOAT)
      , _partial_qscale(_weight.device()) {
    }

    void Dense::mask_weights(const StorageView& index) {
      ops::Gather()(_weight, index, _partial_weight);
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
      const StorageView* qscale = _partial_qscale.empty() ? _qscale : &_partial_qscale;
      const StorageView* weight = _partial_weight.empty() ? &_weight : &_partial_weight;
      const StorageView* bias = _partial_bias.empty() ? _bias : &_partial_bias;

      static const ops::Gemm gemm_op(1, 0, false, false, true);
      if (_weight.dtype() == DataType::DT_INT16 || _weight.dtype() == DataType::DT_INT8) {
        const auto device = input.device();
        StorageView qinput(_weight.dtype(), device);
        StorageView qinput_scale(_qscale->dtype(), device);
        StorageView qoutput(DataType::DT_INT32, device);
        ops::Quantize()(input, qinput, qinput_scale);
        gemm_op(qinput, *weight, *bias, qoutput);
        ops::Dequantize()(qoutput, qinput_scale, *qscale, output);
      } else {
        gemm_op(input, *weight, *bias, output);
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
