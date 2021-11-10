#include "ctranslate2/ops/gather.h"

#include <algorithm>

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    static inline Shape compute_output_shape(const StorageView& data,
                                             const StorageView& input,
                                             const dim_t axis) {
      Shape output_shape(input.shape());
      for (dim_t i = axis + 1; i < data.rank(); ++i)
        output_shape.push_back(data.dim(i));
      return output_shape;
    }

    static bool support_gather_batch_inplace(const StorageView& data, const StorageView& input) {
      // We can gather in place if the output is not larger than data and indices are in
      // strictly increasing order (i.e. we never need to gather from a previous index).
      const auto* input_begin = input.data<int32_t>();
      const auto* input_end = input_begin + input.size();
      return (input.device() == Device::CPU
              && input.size() <= data.dim(0)
              && std::adjacent_find(input_begin, input_end, std::greater_equal<int32_t>()) == input_end);
    }

    template <typename T>
    void gather_batch_inplace(StorageView& data, const StorageView& input) {
      const auto* indices = input.data<int32_t>();
      auto* dst = data.data<T>();
      const auto* src = dst;
      const auto copy_dim = data.stride(0);
      for (dim_t i = 0; i < input.size(); ++i) {
        const dim_t index = indices[i];
        if (index != i)
          primitives<Device::CPU>::copy(src + index * copy_dim, dst, copy_dim);
        dst += copy_dim;
      }
    }


    Gather::Gather(const dim_t axis, const dim_t batch_dims)
      : _axis(axis)
      , _batch_dims(batch_dims) {
    }

    void Gather::operator()(StorageView& data, const StorageView& input) const {
      if (_axis == 0 && _batch_dims == 0 && support_gather_batch_inplace(data, input)) {
        PROFILE("Gather");
        TYPE_DISPATCH(data.dtype(), (gather_batch_inplace<T>(data, input)));
        data.resize(compute_output_shape(data, input, _axis));
      } else {
        StorageView clone(std::move(data));
        operator()(clone, input, data);
      }
    }

    void Gather::operator()(const StorageView& data,
                            const StorageView& input,
                            StorageView& output) const {
      PROFILE("Gather");

      if (_batch_dims > 0) {
        if (data.rank() < _batch_dims)
          throw std::invalid_argument("Gather: rank of data should greater than or equal to "
                                      + std::to_string(_batch_dims));
        if (input.rank() < _batch_dims)
          throw std::invalid_argument("Gather: rank of input should greater than or equal to "
                                      + std::to_string(_batch_dims));

        const auto& data_shape = data.shape();
        const auto& input_shape = input.shape();
        if (!std::equal(data_shape.begin(),
                        data_shape.begin() + _batch_dims,
                        input_shape.begin()))
          throw std::invalid_argument("Gather: first " + std::to_string(_batch_dims)
                                      + " dimensions of data and input should match");
      }

      const dim_t axis = _axis < 0 ? data.rank() + _axis : _axis;
      output.resize(compute_output_shape(data, input, axis));
      DEVICE_AND_TYPE_DISPATCH(data.device(), data.dtype(),
                               (compute<D, T>(data, input, axis, _batch_dims, output)));
    }

  }
}
