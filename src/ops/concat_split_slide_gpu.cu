#include "ctranslate2/ops/concat.h"
#include "ctranslate2/ops/split.h"
#include "ctranslate2/ops/slide.h"

#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scatter.h>

#include "cuda/helpers.h"
#include "cuda/utils.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    // Map indices into a larger output with an offset in the depth dimension.
    template <typename T>
    class depth_offset_map {
    private:
      const T _offset;
      const T _input_depth;
      const T _output_depth;
    public:
      depth_offset_map(const T offset, const T input_depth, const T output_depth)
        : _offset(offset)
        , _input_depth(input_depth)
        , _output_depth(output_depth) {
      }
      __device__
      T operator()(const T i) const {
        const T row = i / _input_depth;
        const T col = i % _input_depth;
        return row * _output_depth + col + _offset;
      }
    };

    // Map indices into a larger output with an offset in an inner dimension.
    template <typename T>
    class inner_dim_offset_map {
    private:
      const T _offset;
      const T _input_dim;
      const T _output_dim;
      const T _inner_size;
    public:
      inner_dim_offset_map(const T offset,
                           const T input_dim,
                           const T output_dim,
                           const T inner_size)
        : _offset(offset)
        , _input_dim(input_dim)
        , _output_dim(output_dim)
        , _inner_size(inner_size) {
      }
      __device__
      T operator()(const T i) const {
        const T i0 = i / (_input_dim * _inner_size);
        const T i1 = (i / _inner_size) % _input_dim;
        const T i2 = i % _inner_size;
        return i0 * (_output_dim * _inner_size) + (i1 + _offset) * _inner_size + i2;
      }
    };

    template <Device D, typename T>
    void Concat::compute(const std::vector<const StorageView*>& inputs,
                         StorageView& output) const {
      const dim_t axis = _axis < 0 ? output.rank() + _axis : _axis;
      const dim_t output_dim = output.dim(axis);
      const dim_t inner_size = output.stride(axis);
      const dim_t inner_bytes = inner_size * sizeof (T);
      T* output_data = output.data<T>();
      dim_t offset = 0;

      for (const StorageView* input : inputs) {
        const T* input_data = input->data<T>();
        const dim_t input_size = input->size();
        const dim_t input_bytes = input_size * sizeof (T);

        if (axis == 0) {
          primitives<D>::copy(input_data, output_data + offset, input_size);
          offset += input_size;

        } else {
          const dim_t input_dim = input->dim(axis);

          if (inner_size == 1) {
            auto map_ids = thrust::make_transform_iterator(
              thrust::counting_iterator<cuda::index_t>(0),
              depth_offset_map<cuda::index_t>(offset, input_dim, output_dim));
            THRUST_CALL(thrust::scatter, input_data, input_data + input_size, map_ids, output_data);
          } else if (inner_bytes % sizeof (uint4) == 0 && input_bytes % sizeof (uint4) == 0) {
            auto map_ids = thrust::make_transform_iterator(
              thrust::counting_iterator<cuda::index_t>(0),
              inner_dim_offset_map<cuda::index_t>(offset,
                                                  input_dim,
                                                  output_dim,
                                                  inner_bytes / sizeof (uint4)));
            THRUST_CALL(thrust::scatter,
                        reinterpret_cast<const uint4*>(input_data),
                        reinterpret_cast<const uint4*>(input_data + input_size),
                        map_ids,
                        reinterpret_cast<uint4*>(output_data));
          } else {
            auto map_ids = thrust::make_transform_iterator(
              thrust::counting_iterator<cuda::index_t>(0),
              inner_dim_offset_map<cuda::index_t>(offset, input_dim, output_dim, inner_size));
            THRUST_CALL(thrust::scatter, input_data, input_data + input_size, map_ids, output_data);
          }

          offset += input_dim;
        }
      }
    }

    template <Device D, typename T>
    void Split::compute(const StorageView& input,
                        std::vector<StorageView*>& outputs) const {
      const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      const dim_t input_dim = input.dim(axis);
      const dim_t inner_size = input.stride(axis);
      const dim_t inner_bytes = inner_size * sizeof (T);
      const T* input_data = input.data<T>();
      dim_t offset = 0;

      for (StorageView* output : outputs) {
        T* output_data = output->data<T>();
        const dim_t output_size = output->size();
        const dim_t output_bytes = output_size * sizeof (T);

        if (axis == 0) {
          primitives<D>::copy(input_data + offset, output_data, output_size);
          offset += output_size;

        } else {
          const dim_t output_dim = output->dim(axis);

          if (inner_size == 1) {
            auto map_ids = thrust::make_transform_iterator(
              thrust::counting_iterator<cuda::index_t>(0),
              depth_offset_map<cuda::index_t>(offset, output_dim, input_dim));
            THRUST_CALL(thrust::gather, map_ids, map_ids + output_size, input_data, output_data);
          } else if (inner_bytes % sizeof (uint4) == 0 && output_bytes % sizeof (uint4) == 0) {
            auto map_ids = thrust::make_transform_iterator(
              thrust::counting_iterator<cuda::index_t>(0),
              inner_dim_offset_map<cuda::index_t>(offset,
                                                  output_dim,
                                                  input_dim,
                                                  inner_bytes / sizeof (uint4)));
            THRUST_CALL(thrust::gather,
                        map_ids,
                        map_ids + output_bytes / sizeof (uint4),
                        reinterpret_cast<const uint4*>(input_data),
                        reinterpret_cast<uint4*>(output_data));
          } else {
            auto map_ids = thrust::make_transform_iterator(
              thrust::counting_iterator<cuda::index_t>(0),
              inner_dim_offset_map<cuda::index_t>(offset, output_dim, input_dim, inner_size));
            THRUST_CALL(thrust::gather, map_ids, map_ids + output_size, input_data, output_data);
          }

          offset += output_dim;
        }
      }
    }

    template <Device D, typename T>
    void Slide::compute(const StorageView& input, StorageView& output, const dim_t& index) const {
      const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      const dim_t input_dim = input.dim(axis);
      const dim_t inner_size = input.stride(axis) == 0 ? 1 : input.stride(axis);
      const dim_t inner_bytes = inner_size * sizeof (T);
      const T* input_data = input.data<T>();

      T* output_data = output.data<T>();
      const dim_t output_size = output.size();
      const dim_t output_bytes = output_size * sizeof (T);
      if (axis == 0) {
        dim_t offset = index * output.stride(axis);
        primitives<D>::copy(input_data + offset, output_data, output_size);
      }
      else {
        const dim_t output_dim = output.dim(axis);

        if (inner_size == 1) {
          auto map_ids = thrust::make_transform_iterator(
            thrust::counting_iterator<cuda::index_t>(0),
            depth_offset_map<cuda::index_t>(index, output_dim, input_dim));
          THRUST_CALL(thrust::gather, map_ids, map_ids + output_size, input_data, output_data);
        } else if (inner_bytes % sizeof(uint4) == 0 && output_bytes % sizeof(uint4) == 0) {
          auto map_ids = thrust::make_transform_iterator(
            thrust::counting_iterator<cuda::index_t>(0),
            inner_dim_offset_map<cuda::index_t>(index,
                                                output_dim,
                                                input_dim,
                                                inner_bytes / sizeof(uint4)));
          THRUST_CALL(thrust::gather,
                      map_ids,
                      map_ids + output_bytes / sizeof(uint4),
                      reinterpret_cast<const uint4 *>(input_data),
                      reinterpret_cast<uint4 *>(output_data));
        } else {
          auto map_ids = thrust::make_transform_iterator(
            thrust::counting_iterator<cuda::index_t>(0),
            inner_dim_offset_map<cuda::index_t>(index, output_dim, input_dim, inner_size));
          THRUST_CALL(thrust::gather, map_ids, map_ids + output_size, input_data, output_data);
        }
      }
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Concat::compute<Device::CUDA, T>(const std::vector<const StorageView*>& inputs, \
                                     StorageView& output) const;        \
    template void                                                       \
    Split::compute<Device::CUDA, T>(const StorageView& input,           \
                                    std::vector<StorageView*>& outputs) const;      \
    template void                                                       \
    Slide::compute<Device::CUDA, T>(const StorageView& input,           \
                                    StorageView& output, const dim_t& index) const;
    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
