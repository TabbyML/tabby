#include "ctranslate2/ops/alibi_add.h"

#include "cpu/parallel.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void AlibiAdd::compute(const StorageView& input,
                           const StorageView& alibi,
                           const dim_t alibi_offset,
                           StorageView& output) const {
      const dim_t batch_size = input.dim(0);
      const dim_t num_heads = input.dim(1);
      const dim_t query_length = input.dim(2);
      const dim_t key_length = input.dim(3);

      cpu::parallel_for(0, batch_size * num_heads, 1, [&](const dim_t begin, const dim_t end) {
        for (dim_t i = begin; i < end; ++i) {
          const dim_t b = i / num_heads;
          const dim_t h = i % num_heads;

          for (dim_t q = 0; q < query_length; ++q) {
            primitives<Device::CPU>::add(input.index<float>({b, h, q, 0}),
                                         alibi.index<float>({0, h, 0, alibi_offset}),
                                         output.index<float>({b, h, q, 0}),
                                         key_length);
          }
        }
      });
    }

#define DECLARE_IMPL(T)                                          \
    template void                                                \
    AlibiAdd::compute<Device::CPU, T>(const StorageView& input,  \
                                      const StorageView& alibi,  \
                                      const dim_t alibi_offset,  \
                                      StorageView& output) const;

    DECLARE_IMPL(float)

  }
}
