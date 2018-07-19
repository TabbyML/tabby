#pragma once

#include <memory>

#include "vocabulary.h"
#include "encoder.h"
#include "decoder.h"

namespace ctranslate2 {

  // Base class for models.
  class Model {
  public:
    virtual ~Model() = default;

    virtual const Vocabulary& get_source_vocabulary() const = 0;
    virtual const Vocabulary& get_target_vocabulary() const = 0;

    // Makes new graph to execute this model. Graphs returned by these function
    // should support being executed in parallel without duplicating the model
    // data (i.e. the weights).
    virtual std::unique_ptr<Encoder> make_encoder() const = 0;
    virtual std::unique_ptr<Decoder> make_decoder() const = 0;

    static StorageView load_data(const Shape& shape, size_t data_width, void* data);
  };


  enum class ModelType {
    Transformer
  };


  // Model factory from a path.
  class ModelFactory {
  public:
    static std::shared_ptr<Model> load(const std::string& type, const std::string& path);
    static std::shared_ptr<Model> load(ModelType type, const std::string& path);
  };

}
