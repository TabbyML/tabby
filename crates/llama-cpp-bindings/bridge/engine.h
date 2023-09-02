#pragma once

#include "rust/cxx.h"
#include <memory>

namespace tabby {

class LlamaEngine {
 public:
  virtual ~LlamaEngine();
  virtual rust::Vec<uint32_t> inference(
      rust::Slice<const rust::String> tokens,
      size_t max_decoding_length,
      float sampling_temperature
  ) const = 0;
};

std::shared_ptr<LlamaEngine> create_engine(rust::Str model_path);
}  // namespace
