#pragma once

#include "rust/cxx.h"
#include <memory>

namespace tabby {

class TextInferenceEngine {
 public:
  virtual ~TextInferenceEngine();
  virtual rust::Vec<uint32_t> inference(
      const rust::Str prompt,
      size_t max_decoding_length,
      float sampling_temperature
  ) const = 0;
};

std::shared_ptr<TextInferenceEngine> create_engine(rust::Str model_path);
}  // namespace
