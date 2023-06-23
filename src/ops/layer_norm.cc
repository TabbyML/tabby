#include "ctranslate2/ops/layer_norm.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    LayerNorm::LayerNorm(const dim_t axis, const float epsilon)
      : _axis(axis)
      , _epsilon(epsilon)
    {
    }

    void LayerNorm::operator()(const StorageView& beta,
                               const StorageView& gamma,
                               const StorageView& input,
                               StorageView& output) const {
      operator()(&beta, &gamma, input, output);
    }

    void LayerNorm::operator()(StorageView& input) const {
      operator()(input, input);
    }

    void LayerNorm::operator()(const StorageView& input, StorageView& output) const {
      operator()(nullptr, nullptr, input, output);
    }

    void LayerNorm::operator()(const StorageView* beta,
                               const StorageView* gamma,
                               const StorageView& input,
                               StorageView& output) const {
      PROFILE("LayerNorm");
      output.resize_as(input);

      const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      const dim_t axis_size = input.dim(axis);

      dim_t inner_size = 1;
      dim_t outer_size = 1;
      for (dim_t i = 0; i < axis; ++i)
        outer_size *= input.dim(i);
      for (dim_t i = axis + 1; i < input.rank(); ++i)
        inner_size *= input.dim(i);

      DEVICE_AND_FLOAT_DISPATCH("LayerNorm", input.device(), input.dtype(),
                                (compute<D, T>(beta,
                                               gamma,
                                               input,
                                               axis,
                                               outer_size,
                                               axis_size,
                                               inner_size,
                                               output)));
    }

  }
}
