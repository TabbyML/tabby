#pragma once

#include <string>
#include <vector>

#include "model.h"
#include "vocabulary_map.h"

namespace opennmt {

  class Translator {
  public:
    Translator(const std::shared_ptr<Model>& model,
               size_t max_decoding_steps,
               size_t beam_size,
               float length_penalty,
               const std::string& vocabulary_map);
    Translator(const Translator& other);

    std::vector<std::string>
    translate(const std::vector<std::string>& tokens);

    std::vector<std::vector<std::string>>
    translate_batch(const std::vector<std::vector<std::string>>& tokens);

  private:
    const std::shared_ptr<Model> _model;
    std::shared_ptr<VocabularyMap> _vocabulary_map;
    std::unique_ptr<Encoder> _encoder;
    std::unique_ptr<Decoder> _decoder;
    size_t _max_decoding_steps;
    size_t _beam_size;
    float _length_penalty;
  };

}
