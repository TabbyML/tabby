#include "ctranslate2/ops/concat.h"
#include "ctranslate2/ops/split.h"
#include "ctranslate2/ops/slide.h"

#include "cpu/parallel.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    static dim_t compute_copy_size(const StorageView& x, dim_t axis) {
      dim_t copy_size = 1;
      for (dim_t i = axis; i < x.rank(); ++i)
        copy_size *= x.dim(i);
      return copy_size;
    }

    static dim_t compute_iter_size(const StorageView& x, dim_t axis) {
      dim_t iter_size = 1;
      for (dim_t i = 0; i < axis; ++i)
        iter_size *= x.dim(i);
      return iter_size;
    }

    template <Device D, typename T>
    void Concat::compute(const std::vector<const StorageView*>& inputs,
                         StorageView& output) const {
      const dim_t axis = _axis < 0 ? output.rank() + _axis : _axis;
      const dim_t step_size = output.dim(axis) * output.stride(axis);
      T* output_data = output.data<T>();

      for (const StorageView* input : inputs) {
        const StorageView& x = *input;
        const dim_t copy_size = compute_copy_size(x, axis);
        if (copy_size == 0)
          continue;
        const dim_t iter_size = compute_iter_size(x, axis);
        const T* x_data = x.data<T>();

        const dim_t grain_size = cpu::get_minimum_batch_copies_per_thread<T>(copy_size);
        cpu::parallel_for(0, iter_size, grain_size, [&](dim_t begin, dim_t end) {
          for (dim_t i = begin; i < end; ++i)
            primitives<D>::copy(x_data + i * copy_size, output_data + i * step_size, copy_size);
        });

        output_data += copy_size;  // Copy next input with an offset.
      }
    }

    template <Device D, typename T>
    void Split::compute(const StorageView& input,
                        std::vector<StorageView*>& outputs) const {
      const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      const dim_t step_size = input.dim(axis) * input.stride(axis);
      const T* input_data = input.data<T>();

      for (StorageView* output : outputs) {
        StorageView& x = *output;
        const dim_t copy_size = compute_copy_size(x, axis);
        if (copy_size == 0)
          continue;
        const dim_t iter_size = compute_iter_size(x, axis);
        T* x_data = x.data<T>();

        const dim_t grain_size = cpu::get_minimum_batch_copies_per_thread<T>(copy_size);
        cpu::parallel_for(0, iter_size, grain_size, [&](dim_t begin, dim_t end) {
          for (dim_t i = begin; i < end; ++i)
            primitives<D>::copy(input_data + i * step_size, x_data + i * copy_size, copy_size);
        });

        input_data += copy_size;  // Read next with an offset.
      }
    }

    template <Device D, typename T>
    void Slide::compute(const StorageView& input, StorageView& output, const dim_t& index) const {
      const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      const dim_t stride_axis = input.stride(axis) == 0 ? 1 : input.stride(axis);
      const dim_t step_size = input.dim(axis) * stride_axis;
      const T* input_data = input.data<T>();

      StorageView& x = output;
      T* x_data = x.data<T>();

      const dim_t copy_size = compute_copy_size(x, axis);
      if (copy_size == 0)
        return;

      const dim_t iter_size = compute_iter_size(x, axis);

      const dim_t grain_size = cpu::get_minimum_batch_copies_per_thread<T>(copy_size);
      input_data += index * stride_axis;  // Read next with an offset.
      cpu::parallel_for(0, iter_size, grain_size, [&](dim_t begin, dim_t end) {
        for (dim_t i = begin; i < end; ++i)
          primitives<D>::copy(input_data + i * step_size, x_data + i * copy_size, copy_size);
      });
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Concat::compute<Device::CPU, T>(const std::vector<const StorageView*>& inputs, \
                                    StorageView& output) const;         \
    template void                                                       \
    Split::compute<Device::CPU, T>(const StorageView& input,            \
                                   std::vector<StorageView*>& outputs) const; \
    template void                                                       \
    Slide::compute<Device::CPU, T>(const StorageView& input,            \
                                   StorageView& output,                 \
                                   const dim_t& index) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
