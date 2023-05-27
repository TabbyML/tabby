#pragma once

#include "rust/cxx.h"
#include <memory>

namespace tabby {

class TextInferenceEngine {
 public:
  virtual ~TextInferenceEngine();
  virtual rust::Vec<rust::String> inference(
      rust::Slice<const rust::String> tokens,
      size_t max_decoding_length,
      float sampling_temperature,
      size_t beam_size
  ) const = 0;
};

std::unique_ptr<TextInferenceEngine> create_engine(
    rust::Str model_path,
    rust::Str model_type,
    rust::Str device,
    rust::Slice<const int32_t> device_indices,
    size_t num_replicas_per_device
);
}  // namespace
