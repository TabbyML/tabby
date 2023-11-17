#include "ctranslate2/ops/slide.h"

#include <numeric>

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    Slide::Slide(dim_t axis, const dim_t& index, const dim_t& size, bool no_copy)
      : _axis(axis)
      , _index(index)
      , _size(size)
      , _no_copy(no_copy) {
      check_arguments();
    }

    void Slide::operator()(const StorageView& input, StorageView& output) const {
      PROFILE("Slide");
      const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;

      if (_index < 0 || _index >= input.dim(axis))
        throw std::invalid_argument("Index or Size given is not valid");

      dim_t offset = input.stride(0) * _index;
      auto shape = input.shape();
      shape[axis] = _size;
      if (_no_copy) {
        TYPE_DISPATCH(input.dtype(),
                      output.view(const_cast<T*>(input.data<T>() + offset), std::move(shape)));
      }
      else {
        output.resize(std::move(shape));
      }

      if (!_no_copy) {
        DEVICE_AND_TYPE_DISPATCH(input.device(), input.dtype(), (compute<D, T>(input, output, _index)));
      }
    }

    void Slide::check_arguments() const {
      if (_no_copy && _axis != 0)
        throw std::invalid_argument("no_copy is only defined when splitting across the first dimension");
    }

  }
}
