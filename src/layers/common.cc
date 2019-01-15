#include "ctranslate2/layers/common.h"

namespace ctranslate2 {
  namespace layers {

    Dense::Dense(const models::Model& model, const std::string& scope)
      : _weight(model.get_variable(scope + "/weight"))
      , _bias(model.get_variable(scope + "/bias"))
      , _qscale(model.get_variable_if_exists(scope + "/weight_scale"))
      , _partial_weight(_weight.device(), _weight.dtype())
      , _partial_bias(_bias.device(), _bias.dtype()) {
    }

    void Dense::operator()(const StorageView& input,
                           StorageView& output,
                           const StorageView* index) {
      const StorageView* weight = &_weight;
      const StorageView* bias = &_bias;
      if (index && !index->empty()) {
        ops::Gather()(_weight, *index, _partial_weight);
        ops::Gather()(_bias, *index, _partial_bias);
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
