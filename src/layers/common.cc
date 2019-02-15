#include "ctranslate2/layers/common.h"

namespace ctranslate2 {
  namespace layers {

    Embeddings::Embeddings(const models::Model& model, const std::string& scope)
      : _embeddings(model.get_variable(scope + "/weight"))
      , _qscale(model.get_variable_if_exists(scope + "/weight_scale")) {
    }

    void Embeddings::operator()(const StorageView& ids,
                                StorageView& output) {
      if (_embeddings.dtype() == DataType::DT_INT16) {
        StorageView gathered(_embeddings.dtype());
        _gather_op(_embeddings, ids, gathered);
        ops::Unquantize()(gathered, *_qscale, output);
      } else {
        _gather_op(_embeddings, ids, output);
      }
    }


    Dense::Dense(const models::Model& model, const std::string& scope)
      : _weight(model.get_variable(scope + "/weight"))
      , _bias(model.get_variable(scope + "/bias"))
      , _qscale(model.get_variable_if_exists(scope + "/weight_scale"))
      , _partial_weight(_weight.device(), _weight.dtype())
      , _partial_bias(_bias.device(), _bias.dtype()) {
    }

    void Dense::mask_weights(const StorageView& index) {
      ops::Gather()(_weight, index, _partial_weight);
      ops::Gather()(_bias, index, _partial_bias);
    }

    void Dense::reset_mask() {
      _partial_weight.clear();
      _partial_bias.clear();
    }

    void Dense::operator()(const StorageView& input, StorageView& output) {
      const StorageView* weight = nullptr;
      const StorageView* bias = nullptr;
      if (_partial_weight.empty()) {
        weight = &_weight;
        bias = &_bias;
      } else {
        weight = &_partial_weight;
        bias = &_partial_bias;
      }

      static const ops::Gemm gemm_op(1, 0, false, false, true);
      if (_weight.dtype() == DataType::DT_FLOAT) {
        gemm_op(input, *weight, *bias, output);
      } else {
        StorageView quantized_input(_weight.dtype());
        StorageView quantized_output(DataType::DT_INT32);
        StorageView squared_scale(_qscale->as_scalar<float>() * _qscale->as_scalar<float>());
        ops::Quantize()(input, *_qscale, quantized_input);
        gemm_op(quantized_input, *weight, *bias, quantized_output);
        ops::Unquantize()(quantized_output, squared_scale, output);
      }

      DEVICE_DISPATCH(output.device(),
                      primitives<D>::add_batch_broadcast(bias->data<float>(),
                                                         output.data<float>(),
                                                         bias->size(),
                                                         output.size()));
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
