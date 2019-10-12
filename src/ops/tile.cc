#include "ctranslate2/ops/tile.h"

namespace ctranslate2 {
  namespace ops {

    void Tile::operator()(const StorageView& input,
                          const StorageView& repeats,
                          StorageView& output) const {
      DEVICE_DISPATCH(input.device(),
                      TYPE_DISPATCH(input.dtype(), (compute<D, T>(input, repeats, output))));
    }

  }
}
