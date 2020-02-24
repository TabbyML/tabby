#include "ctranslate2/ops/tile.h"

#include "../device_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void Tile::operator()(const StorageView& input,
                          const StorageView& repeats,
                          StorageView& output) const {
      PROFILE("Tile");
      DEVICE_DISPATCH(input.device(),
                      TYPE_DISPATCH(input.dtype(), (compute<D, T>(input, repeats, output))));
    }

    void Tile::operator()(StorageView& input, const StorageView& repeats) const {
      StorageView input_clone(std::move(input));
      operator()(input_clone, repeats, input);
    }

  }
}
