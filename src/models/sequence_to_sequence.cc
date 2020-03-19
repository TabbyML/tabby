#include "ctranslate2/models/sequence_to_sequence.h"

namespace ctranslate2 {
  namespace models {

    SequenceToSequenceModel::SequenceToSequenceModel(const std::string& path, size_t spec_revision)
      : Model(path, spec_revision) {
      try {
        _shared_vocabulary.reset(new Vocabulary(path + "/shared_vocabulary.txt"));
      } catch (std::exception&) {
        _source_vocabulary.reset(new Vocabulary(path + "/source_vocabulary.txt"));
        _target_vocabulary.reset(new Vocabulary(path + "/target_vocabulary.txt"));
      }
      _vocabulary_map.reset(new VocabularyMap(path + "/vmap.txt", get_target_vocabulary()));
    }

    const Vocabulary& SequenceToSequenceModel::get_source_vocabulary() const {
      return _shared_vocabulary ? *_shared_vocabulary : *_source_vocabulary;
    }

    const Vocabulary& SequenceToSequenceModel::get_target_vocabulary() const {
      return _shared_vocabulary ? *_shared_vocabulary : *_target_vocabulary;
    }

    const VocabularyMap& SequenceToSequenceModel::get_vocabulary_map() const {
      return *_vocabulary_map;
    }

  }
}
