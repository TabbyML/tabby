#include "ctranslate2/ops/gather.h"

#include <algorithm>

#include "../device_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    static Shape compute_output_shape(const StorageView& data, const StorageView& input) {
      Shape output_shape(input.shape());
      for (size_t i = 1; i < data.rank(); ++i)
        output_shape.push_back(data.dim(i));
      return output_shape;
    }

    static bool support_gather_batch_inplace(const StorageView& data, const StorageView& input) {
      // We can gather in place if the output is not larger than data and indices are in
      // increasing order (i.e. we never need to gather from a previous index).
      return (input.device() == Device::CPU
              && input.size() <= data.dim(0)
              && std::is_sorted(input.data<int32_t>(), input.data<int32_t>() + input.size()));
    }

    template <typename T>
    void gather_batch_inplace(StorageView& data, const StorageView& input) {
      const auto* indices = input.data<int32_t>();
      auto* dst = data.data<T>();
      const auto copy_dim = data.stride(0);
      for (size_t i = 0; i < input.size(); ++i) {
        if (static_cast<size_t>(indices[i]) != i) {
          const auto* src = data.data<T>() + indices[i] * copy_dim;
          primitives<Device::CPU>::copy(src, dst, copy_dim);
        }
        dst += copy_dim;
      }
    }


    Gather::Gather(int axis) {
      if (axis != 0)
        throw std::invalid_argument("unsupported gather axis " + std::to_string(axis));
    }

    void Gather::operator()(StorageView& data, const StorageView& input) const {
      if (support_gather_batch_inplace(data, input)) {
        PROFILE_FUN;
        TYPE_DISPATCH(data.dtype(), (gather_batch_inplace<T>(data, input)));
        data.resize(compute_output_shape(data, input));
      } else {
        StorageView clone(std::move(data));
        operator()(clone, input, data);
      }
    }

    void Gather::operator()(const StorageView& data,
                            const StorageView& input,
                            StorageView& output) const {
      PROFILE_FUN;
      output.resize(compute_output_shape(data, input));
      DEVICE_DISPATCH(data.device(),
                      TYPE_DISPATCH(data.dtype(), (compute<D, T>(data, input, output))));
    }

  }
}
