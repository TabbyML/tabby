#pragma once

#include "ctranslate2/layers/decoder.h"
#include "ctranslate2/layers/encoder.h"
#include "ctranslate2/models/model.h"
#include "ctranslate2/vocabulary.h"
#include "ctranslate2/vocabulary_map.h"

namespace ctranslate2 {
  namespace models {

    class SequenceToSequenceModel : public Model {
    public:
      const Vocabulary& get_source_vocabulary() const;
      const Vocabulary& get_target_vocabulary() const;
      const VocabularyMap* get_vocabulary_map() const;

      // Makes new graph to execute this model. Graphs returned by these function
      // should support being executed in parallel without duplicating the model
      // data (i.e. the weights).
      virtual std::unique_ptr<layers::Encoder> make_encoder() const = 0;
      virtual std::unique_ptr<layers::Decoder> make_decoder() const = 0;

      bool with_source_bos() const {
        return _with_source_bos;
      }

      bool with_source_eos() const {
        return _with_source_eos;
      }

      bool with_target_bos() const {
        return _with_target_bos;
      }

    protected:
      SequenceToSequenceModel(ModelReader& model_reader, size_t spec_revision);
      virtual void finalize() override;

    private:
      std::unique_ptr<const Vocabulary> _source_vocabulary;
      std::unique_ptr<const Vocabulary> _target_vocabulary;
      std::unique_ptr<const Vocabulary> _shared_vocabulary;
      std::unique_ptr<const VocabularyMap> _vocabulary_map;

      bool _with_source_bos = false;
      bool _with_source_eos = false;
      bool _with_target_bos = true;
    };

  }
}
