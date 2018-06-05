#pragma once

#include <vector>

#include "opennmt/storage_view.h"
#include "compute.h"

namespace opennmt {
  namespace ops {

    class Op {
    public:
      virtual ~Op() = default;
      virtual void operator()(const std::vector<StorageView*>& inputs,
                              std::vector<StorageView*>& outputs) const = 0;
    };

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
