#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Slide : public Op {
    public:
      Slide(dim_t axis, const dim_t& index, const dim_t& size, bool no_copy = false);

      void operator()(const StorageView& input, StorageView& output) const;
    private:
      dim_t _axis;
      dim_t _index;
      dim_t _size;
      bool _no_copy;

      void check_arguments() const;

      template <Device D, typename T>
      void compute(const StorageView& input, StorageView& output, const dim_t& index) const;
    };

  }
}
