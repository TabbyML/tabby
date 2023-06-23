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

      DEVICE_AND_FLOAT_DISPATCH("GumbelMax", x.device(), x.dtype(), (add_gumbel_noise<D, T>(x, y)));

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
