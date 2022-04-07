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
      const Vocabulary& get_source_vocabulary() const;
      const Vocabulary& get_target_vocabulary() const;
      const VocabularyMap* get_vocabulary_map() const;

      // Makes new graph to execute this model. Graphs returned by these function
      // should support being executed in parallel without duplicating the model
      // data (i.e. the weights).
      virtual std::unique_ptr<layers::Encoder> make_encoder() const = 0;
      virtual std::unique_ptr<layers::Decoder> make_decoder() const = 0;

      void forward_encoder(layers::Encoder& encoder,
                           const std::vector<std::vector<std::vector<std::string>>>& source,
                           StorageView& memory,
                           StorageView& memory_lengths) const;

      void forward_decoder(layers::Decoder& decoder,
                           layers::DecoderState& state,
                           const std::vector<std::vector<std::string>>& target,
                           StorageView& logits) const;

      void forward(layers::Encoder& encoder,
                   layers::Decoder& decoder,
                   const std::vector<std::vector<std::vector<std::string>>>& source,
                   const std::vector<std::vector<std::string>>& target,
                   StorageView& logits) const;

      std::vector<ScoringResult>
      score(layers::Encoder& encoder,
            layers::Decoder& decoder,
            const std::vector<std::vector<std::string>>& source,
            const std::vector<std::vector<std::string>>& target,
            const ScoringOptions& options = ScoringOptions()) const;

      std::vector<GenerationResult<std::string>>
      translate(layers::Encoder& encoder,
                layers::Decoder& decoder,
                const std::vector<std::vector<std::string>>& source,
                const std::vector<std::vector<std::string>>& target_prefix = {},
                const TranslationOptions& options = TranslationOptions()) const;

    protected:
      virtual void initialize(ModelReader& model_reader) override;

    private:
      std::vector<std::shared_ptr<const Vocabulary>> _source_vocabularies;
      std::shared_ptr<const Vocabulary> _target_vocabulary;
      std::unique_ptr<const VocabularyMap> _vocabulary_map;

      void load_vocabularies(ModelReader& model_reader);
      const std::string* decoder_start_token() const;

      bool _with_source_bos = false;
      bool _with_source_eos = false;
      bool _with_target_bos = true;
      bool _user_decoder_start_tokens = false;
    };

  }
}
