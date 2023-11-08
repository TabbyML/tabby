#pragma once

#include "rust/cxx.h"
#include <memory>

namespace llama {
struct StepOutput;

class TextInferenceEngine {
 public:
  virtual ~TextInferenceEngine();

  virtual void add_request(uint32_t request_id, rust::Str text, size_t max_input_length) = 0;
  virtual void stop_request(uint32_t request_id) = 0;
  virtual rust::Vec<StepOutput> step() = 0;
};

std::unique_ptr<TextInferenceEngine> create_engine(bool use_gpu, rust::Str model_path, uint8_t paralellism);
}  // namespace
