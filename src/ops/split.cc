#include "ctranslate2/ops/split.h"

#include <numeric>
#include <stdexcept>

namespace ctranslate2 {
  namespace ops {

    Split::Split(int axis, bool no_copy)
      : _axis(axis)
      , _total_size(0)
      , _no_copy(no_copy) {
      check_arguments();
    }

    Split::Split(int axis, const std::vector<int>& split, bool no_copy)
      : _axis(axis)
      , _split(split)
      , _total_size(std::accumulate(split.begin(), split.end(), 0))
      , _no_copy(no_copy) {
      check_arguments();
    }

    void Split::operator()(const std::vector<StorageView*>& inputs,
                           std::vector<StorageView*>& outputs) const {
      operator()(*inputs[0], outputs);
    }

    void Split::operator()(const StorageView& input,
                           StorageView& output1,
                           StorageView& output2) const {
      std::vector<StorageView*> outputs{&output1, &output2};
      operator()(input, outputs);
    }

    void Split::operator()(const StorageView& input,
                           StorageView& output1,
                           StorageView& output2,
                           StorageView& output3) const {
      std::vector<StorageView*> outputs{&output1, &output2, &output3};
      operator()(input, outputs);
    }

    void Split::operator()(const StorageView& input, std::vector<StorageView*>& outputs) const {
      auto axis = _axis < 0 ? input.rank() + _axis : _axis;
      auto dim = input.dim(axis);

      if (!_split.empty()) {
        if (_split.size() != outputs.size())
          throw std::invalid_argument(std::to_string(outputs.size())
                                      + " outputs are passed but "
                                      + std::to_string(_split.size())
                                      + " split sizes were configured");
        if (dim != _total_size)
          throw std::invalid_argument("axis " + std::to_string(axis) + " has dimension "
                                      + std::to_string(dim) + " but expected "
                                      + std::to_string(_total_size));

      } else if (dim % outputs.size() != 0)
        throw std::invalid_argument("axis " + std::to_string(axis) + " is not divisble by "
                                    + std::to_string(outputs.size()));

      size_t offset = 0;
      for (size_t j = 0; j < outputs.size(); ++j) {
        auto& x = *outputs[j];
        auto shape = input.shape();
        auto split_size = _split.empty() ? dim / outputs.size() : _split[j];
        shape[axis] = split_size;
        if (_no_copy) {
          TYPE_DISPATCH(input.dtype(),
                        x.view(const_cast<T*>(input.data<T>() + offset), shape));
        } else {
          x.resize(shape);
        }
        offset += input.stride(0) * split_size;
      }

      if (!_no_copy) {
        DEVICE_DISPATCH(input.device(),
                        TYPE_DISPATCH(input.dtype(), (compute<D, T>(input, outputs))));
      }
    }

    void Split::check_arguments() const {
      if (_no_copy && _axis != 0)
        throw std::invalid_argument("no_copy is only defined when splitting across the first dimension");
    }

    template <Device D, typename T>
    void Split::compute(const StorageView& input,
                        std::vector<StorageView*>& outputs) const {
      size_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      size_t offset = 0;
      for (auto* output : outputs) {
        auto& x = *output;
        size_t iter_dim = 1;
        size_t copy_dim = 1;
        for (size_t i = 0; i < axis; ++i)
          iter_dim *= x.dim(i);
        for (size_t i = axis; i < x.rank(); ++i)
          copy_dim *= x.dim(i);
        for (size_t i = 0; i < iter_dim; ++i) {
          primitives<>::copy(input.data<T>() + offset + i * input.dim(axis) * input.stride(axis),
                             x.data<T>() + i * copy_dim,
                             copy_dim);
        }
        offset += copy_dim;
      }
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Split::compute<Device::CPU, T>(const StorageView& input,            \
                                   std::vector<StorageView*>& outputs) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
