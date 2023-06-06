#pragma once

#include "rust/cxx.h"
#include <memory>

namespace tabby {

struct InferenceContext;

typedef rust::Fn<bool(InferenceContext&, size_t, uint32_t, rust::String)> InferenceCallback;

class TextInferenceEngine {
 public:
  virtual ~TextInferenceEngine();
  virtual rust::Vec<uint32_t> inference(
      rust::Box<InferenceContext> context,
      InferenceCallback callback,
      rust::Slice<const rust::String> tokens,
      size_t max_decoding_length,
      float sampling_temperature
  ) const = 0;
};

std::shared_ptr<TextInferenceEngine> create_engine(
    rust::Str model_path,
    rust::Str model_type,
    rust::Str device,
    rust::Slice<const int32_t> device_indices,
    size_t num_replicas_per_device
);
}  // namespace
