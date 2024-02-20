#pragma once

#include "rust/cxx.h"
#include <cmath>
#include <memory>

namespace llama {
struct LlamaInitRequest;
struct StepOutput;

class TextInferenceEngine {
 public:
  virtual ~TextInferenceEngine();

  virtual void add_request(rust::Box<LlamaInitRequest> request) = 0;
  virtual bool has_pending_requests() const = 0;
  virtual void step() = 0;
};

std::unique_ptr<TextInferenceEngine> create_engine(bool use_gpu, rust::Str model_path, uint8_t paralellism);
}  // namespace
