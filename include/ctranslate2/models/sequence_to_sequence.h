#pragma once

#include "ctranslate2/layers/decoder.h"
#include "ctranslate2/layers/encoder.h"
#include "ctranslate2/models/model.h"
#include "ctranslate2/scoring.h"
#include "ctranslate2/translation.h"
#include "ctranslate2/vocabulary.h"
#include "ctranslate2/vocabulary_map.h"

namespace ctranslate2 {
  namespace models {

    class SequenceToSequenceModel : public Model {
    public:
      size_t num_source_vocabularies() const;
      const Vocabulary& get_source_vocabulary(size_t index = 0) const;
      const Vocabulary& get_target_vocabulary() const;
      const VocabularyMap* get_vocabulary_map() const;

      bool with_source_bos() const {
        return _with_source_bos;
      }

      bool with_source_eos() const {
        return _with_source_eos;
      }

      bool with_target_bos() const {
        return _with_target_bos;
      }

      bool user_decoder_start_tokens() const {
        return _user_decoder_start_tokens;
      }

    protected:
      virtual void initialize(ModelReader& model_reader) override;

    private:
      std::vector<std::shared_ptr<const Vocabulary>> _source_vocabularies;
      std::shared_ptr<const Vocabulary> _target_vocabulary;
      std::unique_ptr<const VocabularyMap> _vocabulary_map;

      void load_vocabularies(ModelReader& model_reader);

      bool _with_source_bos = false;
      bool _with_source_eos = false;
      bool _with_target_bos = true;
      bool _user_decoder_start_tokens = false;
    };


    class SequenceToSequenceReplica : public ModelReplica {
    public:
      SequenceToSequenceReplica(const std::shared_ptr<const Model>& model)
        : ModelReplica(model)
      {
      }

      virtual std::vector<ScoringResult>
      score(const std::vector<std::vector<std::string>>& source,
            const std::vector<std::vector<std::string>>& target,
            const ScoringOptions& options = ScoringOptions()) = 0;

      virtual std::vector<TranslationResult>
      translate(const std::vector<std::vector<std::string>>& source,
                const std::vector<std::vector<std::string>>& target_prefix = {},
                const TranslationOptions& options = TranslationOptions()) = 0;
    };


    class EncoderDecoderReplica : public SequenceToSequenceReplica {
    public:
      EncoderDecoderReplica(const std::shared_ptr<const SequenceToSequenceModel>& model,
                            std::unique_ptr<layers::Encoder> encoder,
                            std::unique_ptr<layers::Decoder> decoder);

      layers::Encoder& encoder() {
        return *_encoder;
      }

      layers::Decoder& decoder() {
        return *_decoder;
      }

      std::vector<ScoringResult>
      score(const std::vector<std::vector<std::string>>& source,
            const std::vector<std::vector<std::string>>& target,
            const ScoringOptions& options = ScoringOptions()) override;

      std::vector<TranslationResult>
      translate(const std::vector<std::vector<std::string>>& source,
                const std::vector<std::vector<std::string>>& target_prefix = {},
                const TranslationOptions& options = TranslationOptions()) override;

    private:
      std::vector<std::vector<size_t>>
      make_source_ids(const std::vector<std::vector<std::string>>& source, size_t index) const;
      std::vector<std::vector<size_t>>
      make_target_ids(const std::vector<std::vector<std::string>>& target, bool partial) const;
      bool source_is_empty(const std::vector<std::string>& source) const;

      void encode(const std::vector<std::vector<std::vector<std::string>>>& source,
                  StorageView& memory,
                  StorageView& memory_lengths);

      const std::shared_ptr<const SequenceToSequenceModel> _model;
      const std::unique_ptr<layers::Encoder> _encoder;
      const std::unique_ptr<layers::Decoder> _decoder;
    };

  }
}
