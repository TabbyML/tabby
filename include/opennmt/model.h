#pragma once

#include <memory>

#include "vocabulary.h"
#include "encoder.h"
#include "decoder.h"

namespace opennmt {

  class Model {
  public:
    virtual ~Model() = default;

    virtual const Vocabulary& get_source_vocabulary() const = 0;
    virtual const Vocabulary& get_target_vocabulary() const = 0;

    virtual std::unique_ptr<Encoder> make_encoder() const = 0;
    virtual std::unique_ptr<Decoder> make_decoder() const = 0;
  };


  enum class ModelType {
    Transformer
  };


  class ModelFactory {
  public:
    static std::shared_ptr<Model> load(ModelType type, const std::string& path);
  };

}
