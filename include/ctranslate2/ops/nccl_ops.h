#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {
    class ReduceAll : public Op {
    public:
      enum class RED_OP {
        SUM,
        PROD,
        MIN,
        MAX,
        AVG
      };

      explicit ReduceAll(RED_OP op = RED_OP::SUM);
      void operator()(const StorageView& input, StorageView& output) const;
    private:
      RED_OP _reduce_op;

      template <Device D, typename T>
      void compute(const StorageView& input, StorageView& output) const;
    };

    class GatherAll : public Op {
    public:
      explicit GatherAll();
      void operator()(const StorageView& input, StorageView& output) const;
    private:
      template <Device D, typename T>
      void compute(const StorageView& input, StorageView& output) const;
    };
  }
}