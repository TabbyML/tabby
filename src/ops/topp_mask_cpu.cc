#include "ctranslate2/ops/topp_mask.h"

#include <algorithm>
#include <numeric>

#include "cpu/parallel.h"
#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <typename T>
    static void topp_mask_kernel(const T* x,
                                 const T* probs,
                                 const float p,
                                 const float mask,
                                 const dim_t batch_size,
                                 const dim_t depth,
                                 T* y) {
      cpu::parallel_for(0, batch_size, 1, [&](dim_t begin, dim_t end) {
        for (dim_t i = begin; i < end; ++i) {
          const auto* x_i = x + (i * depth);
          const auto* probs_i = probs + (i * depth);
          auto* y_i = y + (i * depth);

          std::vector<dim_t> ids(depth);
          std::iota(ids.begin(), ids.end(), 0);
          std::sort(ids.begin(), ids.end(), [&probs_i](const dim_t i1, const dim_t i2) {
            return probs_i[i1] > probs_i[i2];
          });

          float total_p = 0;

          for (const auto id : ids) {
            y_i[id] = total_p < p ? x_i[id] : mask;
            total_p += probs_i[id];
          }
        }
      });
    }

    template <Device D, typename T>
    void TopPMask::compute(const StorageView& input,
                           const StorageView& probs,
                           StorageView& output) const {
      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;

      topp_mask_kernel(input.data<T>(),
                       probs.data<T>(),
                       _p,
                       _mask_value,
                       batch_size,
                       depth,
                       output.data<T>());
    }

    template<>
    dim_t TopPMask::max_num_classes<Device::CPU>() {
      return std::numeric_limits<dim_t>::max();
    }

#define DECLARE_IMPL(T)                                                 \
    template void TopPMask::compute<Device::CPU, T>(const StorageView&, \
                                                    const StorageView&, \
                                                    StorageView&) const;

    DECLARE_IMPL(float)

  }
}
