#include "ctranslate2/ops/gumbel_max.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    GumbelMax::GumbelMax(dim_t num_samples)
      : _num_samples(num_samples)
      , _topk_op(num_samples)
    {
    }

    void GumbelMax::operator()(const StorageView& x,
                               StorageView& values,
                               StorageView& indices) const {
      PROFILE("GumbelMax");

      StorageView y(x.shape(), x.dtype(), x.device());

      switch (x.dtype()) {
      case DataType::FLOAT32:
        DEVICE_DISPATCH(x.device(), (add_gumbel_noise<D, float>(x, y)));
        break;
      case DataType::FLOAT16:
        DEVICE_DISPATCH(x.device(), (add_gumbel_noise<D, float16_t>(x, y)));
        break;
      default:
        throw std::invalid_argument("GumbelMax only supports float types");
      }

      _topk_op(y, values, indices);

      Shape output_shape = x.shape();
      output_shape.back() = _num_samples;
      values.reshape(output_shape);
      indices.reshape(output_shape);
    }

    void GumbelMax::operator()(const StorageView& x, StorageView& indices) const {
      StorageView values(x.dtype(), x.device());
      operator()(x, values, indices);
    }

  }
}
