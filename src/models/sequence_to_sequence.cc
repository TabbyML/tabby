#include "ctranslate2/models/sequence_to_sequence.h"

namespace ctranslate2 {
  namespace models {

    static const std::string shared_vocabulary_file = "shared_vocabulary.txt";
    static const std::string source_vocabulary_file = "source_vocabulary.txt";
    static const std::string target_vocabulary_file = "target_vocabulary.txt";
    static const std::string vmap_file = "vmap.txt";

    SequenceToSequenceModel::SequenceToSequenceModel(ModelReader& model_reader, size_t spec_revision)
      : Model(model_reader, spec_revision) {
      {
        auto shared_vocabulary = model_reader.get_file(shared_vocabulary_file);
        if (shared_vocabulary) {
          _shared_vocabulary.reset(new Vocabulary(*shared_vocabulary));
        } else {
          {
            auto source_vocabulary = model_reader.get_required_file(source_vocabulary_file);
            _source_vocabulary.reset(new Vocabulary(*source_vocabulary));
          }
          {
            auto target_vocabulary = model_reader.get_required_file(target_vocabulary_file);
            _target_vocabulary.reset(new Vocabulary(*target_vocabulary));
          }
        }
      }

      {
        auto vmap = model_reader.get_file(vmap_file);
        if (vmap) {
          _vocabulary_map.reset(new VocabularyMap(*vmap, get_target_vocabulary()));
        }
      }
    }

    const Vocabulary& SequenceToSequenceModel::get_source_vocabulary() const {
      return _shared_vocabulary ? *_shared_vocabulary : *_source_vocabulary;
    }

    const Vocabulary& SequenceToSequenceModel::get_target_vocabulary() const {
      return _shared_vocabulary ? *_shared_vocabulary : *_target_vocabulary;
    }

    const VocabularyMap* SequenceToSequenceModel::get_vocabulary_map() const {
      return _vocabulary_map.get();
    }

  }
}
