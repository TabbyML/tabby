#include "opennmt/model.h"

#include "opennmt/transformer.h"

namespace opennmt {

  std::shared_ptr<Model> ModelFactory::load(ModelType type,
                                            const std::string& path,
                                            const std::string& source_vocabulary_path,
                                            const std::string& target_vocabulary_path) {
    Model* model = nullptr;

    switch (type) {
    case ModelType::Transformer:
      model = new TransformerModel(path, source_vocabulary_path, target_vocabulary_path);
      break;
    }

    return std::shared_ptr<Model>(model);
  }

}
