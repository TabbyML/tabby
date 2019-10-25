#pragma once

#include <cassert>
#include <vector>

#include "ctranslate2/storage_view.h"
#include "ctranslate2/profiler.h"
#include "ctranslate2/primitives/primitives.h"

namespace ctranslate2 {
  namespace ops {

    // Base class for operators.
    class Op {
    public:
      virtual ~Op() = default;

      // Shared interface for graph execution.
      virtual void operator()(const std::vector<StorageView*>& inputs,
                              std::vector<StorageView*>& outputs) const = 0;
    };


    // Base classes for common N-ary operators.
    class UnaryOp : public Op {
    public:
      virtual void operator()(const std::vector<StorageView*>& inputs,
                              std::vector<StorageView*>& outputs) const override {
        operator()(*inputs[0], *outputs[0]);
      }
      virtual void operator()(const StorageView&, StorageView&) const = 0;
    };

    class BinaryOp : public Op {
    public:
      virtual void operator()(const std::vector<StorageView*>& inputs,
                              std::vector<StorageView*>& outputs) const override {
        operator()(*inputs[0], *inputs[1], *outputs[0]);
      }
      virtual void operator()(const StorageView&, const StorageView&, StorageView&) const = 0;
    };

    class TernaryOp : public Op {
    public:
      virtual void operator()(const std::vector<StorageView*>& inputs,
                              std::vector<StorageView*>& outputs) const override {
        operator()(*inputs[0], *inputs[1], *inputs[2], *outputs[0]);
      }
      virtual void operator()(const StorageView&,
                              const StorageView&,
                              const StorageView&,
                              StorageView&) const = 0;
    };

  }
}
