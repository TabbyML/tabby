#pragma once

#include "rust/cxx.h"
#include <memory>

namespace llama {

class TextInferenceEngine {
 public:
  virtual ~TextInferenceEngine();

  virtual void add_request(uint32_t request_id, rust::Slice<const uint32_t> input_token_ids) = 0;
  virtual void stop_request(uint32_t request_id) = 0;
  virtual rust::Vec<uint32_t> step() = 0;

  virtual uint32_t eos_token_id() const = 0;
};

std::unique_ptr<TextInferenceEngine> create_engine(bool use_gpu, rust::Str model_path);
}  // namespace
