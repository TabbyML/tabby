#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class SoftMax : public UnaryOp {
    public:
      SoftMax(bool log = false);

      using UnaryOp::operator();
      void operator()(StorageView& x) const;
      void operator()(const StorageView& x, StorageView& y) const override;
      void operator()(const StorageView& x, const StorageView& lengths, StorageView& y) const;
      void operator()(const StorageView& x, const StorageView* lengths, StorageView& y) const;

    private:
      template <Device D, typename T>
      void compute(const StorageView& input, const StorageView* lengths, StorageView& output) const;

      bool _log;
    };


    class LogSoftMax : public SoftMax {
    public:
      LogSoftMax();
    };

  }
}
