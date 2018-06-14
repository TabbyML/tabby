#include "opennmt/model.h"

#include "opennmt/transformer.h"

namespace opennmt {

  std::shared_ptr<Model> ModelFactory::load(ModelType type, const std::string& path) {
    Model* model = nullptr;

    switch (type) {
    case ModelType::Transformer:
      model = new TransformerModel(path);
      break;
    }

    return std::shared_ptr<Model>(model);
  }

}
