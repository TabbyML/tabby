#pragma once

#include "rust/cxx.h"
#include <memory>

namespace llama {

class TextInferenceEngine {
 public:
  virtual ~TextInferenceEngine();

  virtual void start(rust::Slice<const uint32_t> input_token_ids) = 0;
  virtual uint32_t step() = 0;
  virtual void end() = 0;

  virtual uint32_t eos_token() const = 0;
};

std::unique_ptr<TextInferenceEngine> create_engine(rust::Str model_path);
}  // namespace
