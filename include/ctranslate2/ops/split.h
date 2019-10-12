#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Split : public Op {
    public:
      Split(int axis, bool no_copy = false);
      Split(int axis, const std::vector<int>& split, bool no_copy = false);

      void operator()(const std::vector<StorageView*>& inputs,
                      std::vector<StorageView*>& outputs) const override;
      void operator()(const StorageView& input, StorageView& output1, StorageView& output2) const;
      void operator()(const StorageView& input,
                      StorageView& output1, StorageView& output2, StorageView& output3) const;
      void operator()(const StorageView& input,
                      std::vector<StorageView*>& outputs) const;
    private:
      int _axis;
      std::vector<int> _split;
      size_t _total_size;
      bool _no_copy;

      void check_arguments() const;

      template <Device D, typename T>
      void compute(const StorageView& input,
                   std::vector<StorageView*>& outputs) const;
    };

  }
}
