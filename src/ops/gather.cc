#include "ctranslate2/ops/gather.h"

#include "device_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    Gather::Gather(int axis) {
      if (axis != 0)
        throw std::invalid_argument("unsupported gather axis " + std::to_string(axis));
    }

    void Gather::operator()(const StorageView& data,
                            const StorageView& input,
                            StorageView& output) const {
      Shape output_shape(input.shape());
      for (size_t i = 1; i < data.rank(); ++i)
        output_shape.push_back(data.dim(i));
      output.resize(output_shape);
      DEVICE_DISPATCH(data.device(),
                      TYPE_DISPATCH(data.dtype(), (compute<D, T>(data, input, output))));
    }

    template <Device D, typename T>
    void Gather::compute(const StorageView& data,
                         const StorageView& input,
                         StorageView& output) const {
      const auto* indices = input.data<int32_t>();
      size_t copy_dim = data.stride(0);
      for (size_t i = 0, merge = 1; i < input.size(); i += merge, merge = 1) {
        size_t index = indices[i];
        while (i + merge < input.size() && static_cast<size_t>(indices[i + merge]) == index + merge)
          ++merge;
        const auto* src = data.index<T>({index});
        auto* dst = output.data<T>() + i * copy_dim;
        primitives<Device::CPU>::copy(src, dst, copy_dim * merge);
      }
    }

#define DECLARE_IMPL(T)                                         \
    template void                                               \
    Gather::compute<Device::CPU, T>(const StorageView& data,    \
                                    const StorageView& input,   \
                                    StorageView& output) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
