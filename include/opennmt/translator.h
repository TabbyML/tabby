#pragma once

#include <string>
#include <vector>

#include "model.h"

namespace opennmt {

  class Translator {
  public:
    Translator(const Model& model,
               size_t max_decoding_steps,
               size_t beam_size,
               float length_penalty);

    std::vector<std::string>
    translate(const std::vector<std::string>& tokens);

    std::vector<std::vector<std::string>>
    translate_batch(const std::vector<std::vector<std::string>>& tokens);

  private:
    const Model& _model;
    std::unique_ptr<Encoder> _encoder;
    std::unique_ptr<Decoder> _decoder;
    size_t _max_decoding_steps;
    size_t _beam_size;
    float _length_penalty;
  };

}
