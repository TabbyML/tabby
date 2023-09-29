#pragma once

#include "rust/cxx.h"
#include <memory>

namespace llama {

class TextInferenceEngine {
 public:
  virtual ~TextInferenceEngine();

  virtual void start(rust::Slice<const uint32_t> input_token_ids) const = 0;
  virtual uint32_t step() const = 0;
  virtual void end() const = 0;

  virtual uint32_t eos_token() const = 0;
};

std::shared_ptr<TextInferenceEngine> create_engine(rust::Str model_path);
}  // namespace
