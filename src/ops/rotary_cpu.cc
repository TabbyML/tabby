#include "ctranslate2/ops/rotary.h"

#include "cpu/parallel.h"

namespace ctranslate2 {
  namespace ops {

    template <typename T, bool interleave>
    void rotary_kernel(const T* input,
                       const T* sin,
                       const T* cos,
                       T* output,
                       const dim_t batch_size,
                       const dim_t max_time,
                       const dim_t ndims,
                       const dim_t depth) {
      const dim_t middle = ndims / 2;

      cpu::parallel_for(0, batch_size, 1, [&](dim_t begin, dim_t end) {
        for (dim_t b = begin; b < end; ++b) {
          for (dim_t t = 0; t < max_time; ++t) {
            const T* s = sin + t * ndims;
            const T* c = cos + t * ndims;

            const T* x = input + b * (max_time * depth) + t * depth;
            T* y = output + b * (max_time * depth) + t * depth;

            for (dim_t i = 0; i < ndims; ++i) {
              if (interleave)
                y[i] = x[i] * c[i] + (i % 2 == 0 ? -x[i + 1] : x[i - 1]) * s[i];
              else
                y[i] = x[i] * c[i] + (i < middle ? -x[i + middle] : x[i - middle]) * s[i];
            }

            if (ndims < depth)
              std::copy(x + ndims, x + depth, y + ndims);
          }
        }
      });
    }

    template <Device D, typename T>
    void Rotary::compute(const StorageView& input,
                         const StorageView& sin,
                         const StorageView& cos,
                         StorageView& output,
                         bool is_transposed) const {
      const dim_t max_time = is_transposed ? input.dim(-2) : input.dim(-3);
      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / (max_time * depth);
      const dim_t ndims = _ndims == 0 ? depth : _ndims;

      const auto* x = input.data<T>();
      const auto* s = sin.data<T>();
      const auto* c = cos.data<T>();
      auto* y = output.data<T>();

      if (_interleave)
        rotary_kernel<T, true>(x, s, c, y, batch_size, max_time, ndims, depth);
      else
        rotary_kernel<T, false>(x, s, c, y, batch_size, max_time, ndims, depth);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Rotary::compute<Device::CPU, T>(const StorageView&,                 \
                                    const StorageView&,                 \
                                    const StorageView&,                 \
                                    StorageView&,                       \
                                    bool) const;

    DECLARE_IMPL(float)

  }
}
