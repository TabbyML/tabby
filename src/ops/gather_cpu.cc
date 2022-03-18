#include "ctranslate2/ops/gather.h"

#include "cpu/parallel.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Gather::compute(const StorageView& data,
                         const StorageView& input,
                         const dim_t axis,
                         const dim_t batch_dims,
                         StorageView& output) const {
      const auto* indices = input.data<int32_t>();
      const T* src = data.data<T>();
      T* dst = output.data<T>();

      if (axis == 0 && batch_dims == 0) {
        const dim_t copy_size = data.stride(0);
        const dim_t num_indices = input.size();

        cpu::parallel_for(0, num_indices, 1, [&](dim_t begin, dim_t end) {
          for (dim_t i = begin; i < end; ++i) {
            const int32_t read_index = indices[i];
            primitives<Device::CPU>::copy(src + read_index * copy_size,
                                          dst + i * copy_size,
                                          copy_size);
          }
        });

      } else if (axis == data.rank() - 1 && batch_dims == data.rank() - 1) {
        const dim_t depth = data.dim(-1);
        const dim_t batch_size = data.size() / depth;
        const dim_t gather_size = input.size() / batch_size;  // Num. elements to gather per batch.

        cpu::parallel_for(0, batch_size, 1, [&](dim_t begin, dim_t end) {
          for (dim_t i = begin; i < end; ++i) {
            const auto* indices_row = indices + i * gather_size;
            const T* data_row = src + i * depth;
            T* output_row = dst + i * gather_size;

            for (dim_t j = 0; j < gather_size; ++j)
              output_row[j] = data_row[indices_row[j]];
          }
        });

      } else {
        throw std::invalid_argument("unsupported gather configuration");
      }
    }

#define DECLARE_IMPL(T)                                         \
    template void                                               \
    Gather::compute<Device::CPU, T>(const StorageView& data,    \
                                    const StorageView& input,   \
                                    const dim_t axis,           \
                                    const dim_t batch_dims,     \
                                    StorageView& output) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
