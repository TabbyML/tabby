#pragma once

#include "rust/cxx.h"

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

std::unique_ptr<TextInferenceEngine> create_engine(rust::Str model_path);
}  // namespace
