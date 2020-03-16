#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Split : public Op {
    public:
      Split(dim_t axis, bool no_copy = false);
      Split(dim_t axis, const std::vector<dim_t>& split, bool no_copy = false);

      void operator()(const StorageView& input, StorageView& output1, StorageView& output2) const;
      void operator()(const StorageView& input,
                      StorageView& output1, StorageView& output2, StorageView& output3) const;
      void operator()(const StorageView& input,
                      std::vector<StorageView*>& outputs) const;
    private:
      dim_t _axis;
      std::vector<dim_t> _split;
      dim_t _total_size;
      bool _no_copy;

      void check_arguments() const;

      template <Device D, typename T>
      void compute(const StorageView& input,
                   std::vector<StorageView*>& outputs) const;
    };

  }
}
