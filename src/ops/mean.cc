#include "ctranslate2/ops/mean.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    Mean::Mean(const dim_t axis)
      : _axis(axis)
    {
    }

    void Mean::operator()(const StorageView& input, StorageView& output) const {
      PROFILE("Mean");

      const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      if (axis >= input.rank())
        throw std::out_of_range("Cannot compute mean of axis " + std::to_string(axis)
                                + " for input with rank " + std::to_string(input.rank()));

      const dim_t axis_size = input.dim(axis);
      if (axis_size == 1) {
        output = input;
        output.squeeze(axis);
        return;
      }

      {
        Shape output_shape(input.shape());
        output_shape.erase(output_shape.begin() + axis);
        output.resize(std::move(output_shape));
      }

      dim_t inner_size = 1;
      dim_t outer_size = 1;
      for (dim_t i = 0; i < axis; ++i)
        outer_size *= input.dim(i);
      for (dim_t i = axis + 1; i < input.rank(); ++i)
        inner_size *= input.dim(i);

      switch (input.dtype()) {
      case DataType::FLOAT32: {
        DEVICE_DISPATCH(input.device(),
                        (compute<D, float>(input, outer_size, axis_size, inner_size, output)));
        break;
      }
#ifdef CT2_WITH_CUDA
      case DataType::FLOAT16: {
        if (input.device() != Device::CUDA)
          throw std::invalid_argument("FP16 Mean is only supported on GPU");
        compute<Device::CUDA, float16_t>(input, outer_size, axis_size, inner_size, output);
        break;
      }
#endif
      default:
        throw std::invalid_argument("Mean only supports float (or float16 on GPU)");
      }
    }

  }
}
