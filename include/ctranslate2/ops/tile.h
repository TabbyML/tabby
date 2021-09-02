#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Tile : public UnaryOp {
    public:
      Tile(const dim_t axis, const dim_t num_tiles);

      void operator()(const StorageView& input, StorageView& output) const override;
      void operator()(StorageView& input) const;

    private:
      template <Device D, typename T>
      void compute(const StorageView& input,
                   const dim_t outer_size,
                   const dim_t inner_size,
                   StorageView& output) const;

      const dim_t _axis;
      const dim_t _num_tiles;
    };

  }
}
