#pragma once

#include <cassert>
#include <vector>

#include "ctranslate2/storage_view.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/profiler.h"

namespace ctranslate2 {
  namespace ops {

    // Base class for operators.
    class Op {
    public:
      virtual ~Op() = default;
    };


    // Base classes for common N-ary operators.
    class UnaryOp : public Op {
    public:
      virtual void operator()(const StorageView&, StorageView&) const = 0;
    };

    class BinaryOp : public Op {
    public:
      virtual void operator()(const StorageView&, const StorageView&, StorageView&) const = 0;
    };

    class TernaryOp : public Op {
    public:
      virtual void operator()(const StorageView&,
                              const StorageView&,
                              const StorageView&,
                              StorageView&) const = 0;
    };

  }
}
