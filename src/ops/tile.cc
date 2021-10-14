#include "ctranslate2/ops/tile.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    Tile::Tile(const dim_t axis, const dim_t num_tiles)
      : _axis(axis)
      , _num_tiles(num_tiles)
    {
    }

    void Tile::operator()(const StorageView& input, StorageView& output) const {
      PROFILE("Tile");

      const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      if (axis >= input.rank())
        throw std::out_of_range("Can't tile axis " + std::to_string(axis)
                                + " for input with rank " + std::to_string(input.rank()));

      {
        Shape output_shape(input.shape());
        output_shape[axis] *= _num_tiles;
        output.resize(std::move(output_shape));
      }

      dim_t inner_size = 1;
      dim_t outer_size = 1;
      for (dim_t i = 0; i < axis; ++i)
        outer_size *= input.dim(i);
      for (dim_t i = axis; i < input.rank(); ++i)
        inner_size *= input.dim(i);

      DEVICE_AND_TYPE_DISPATCH(input.device(), input.dtype(),
                               (compute<D, T>(input, outer_size, inner_size, output)));
    }

    void Tile::operator()(StorageView& input) const {
      StorageView input_clone(std::move(input));
      operator()(input_clone, input);
    }

  }
}
