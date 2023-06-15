#pragma once

#include "ctranslate2/layers/decoder.h"
#include "ctranslate2/layers/encoder.h"
#include "ctranslate2/models/model.h"
#include "ctranslate2/scoring.h"
#include "ctranslate2/translation.h"
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
        return config["add_source_bos"];
      }

      bool with_source_eos() const {
        return config["add_source_eos"];
      }

      const std::string* decoder_start_token() const {
        auto& start_token = config["decoder_start_token"];
        return start_token.is_null() ? nullptr : start_token.get_ptr<const std::string*>();
      }

    protected:
      virtual void initialize(ModelReader& model_reader) override;

    private:
      std::vector<std::shared_ptr<const Vocabulary>> _source_vocabularies;
      std::shared_ptr<const Vocabulary> _target_vocabulary;
      std::shared_ptr<const VocabularyMap> _vocabulary_map;

      void load_vocabularies(ModelReader& model_reader);
    };


    class SequenceToSequenceReplica : public ModelReplica {
    public:
      SequenceToSequenceReplica(const std::shared_ptr<const Model>& model)
        : ModelReplica(model)
      {
      }

      static std::unique_ptr<SequenceToSequenceReplica> create_from_model(const Model& model) {
        return model.as_sequence_to_sequence();
      }

      std::vector<ScoringResult>
      score(const std::vector<std::vector<std::string>>& source,
            const std::vector<std::vector<std::string>>& target,
            const ScoringOptions& options = ScoringOptions());

      std::vector<TranslationResult>
      translate(const std::vector<std::vector<std::string>>& source,
                const std::vector<std::vector<std::string>>& target_prefix = {},
                const TranslationOptions& options = TranslationOptions());

    protected:
      virtual bool skip_scoring(const std::vector<std::string>& source,
                                const std::vector<std::string>& target,
                                const ScoringOptions& options,
                                ScoringResult& result) {
        (void)source;
        (void)target;
        (void)options;
        (void)result;
        return false;
      }

      virtual bool skip_translation(const std::vector<std::string>& source,
                                    const std::vector<std::string>& target_prefix,
                                    const TranslationOptions& options,
                                    TranslationResult& result) {
        (void)source;
        (void)target_prefix;
        (void)options;
        (void)result;
        return false;
      }

      virtual std::vector<ScoringResult>
      run_scoring(const std::vector<std::vector<std::string>>& source,
                  const std::vector<std::vector<std::string>>& target,
                  const ScoringOptions& options) = 0;

      virtual std::vector<TranslationResult>
      run_translation(const std::vector<std::vector<std::string>>& source,
                      const std::vector<std::vector<std::string>>& target_prefix,
                      const TranslationOptions& options) = 0;
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

    protected:
      bool skip_scoring(const std::vector<std::string>& source,
                        const std::vector<std::string>& target,
                        const ScoringOptions& options,
                        ScoringResult& result) override;

      bool skip_translation(const std::vector<std::string>& source,
                            const std::vector<std::string>& target_prefix,
                            const TranslationOptions& options,
                            TranslationResult& result) override;

      std::vector<ScoringResult>
      run_scoring(const std::vector<std::vector<std::string>>& source,
                  const std::vector<std::vector<std::string>>& target,
                  const ScoringOptions& options) override;

      std::vector<TranslationResult>
      run_translation(const std::vector<std::vector<std::string>>& source,
                      const std::vector<std::vector<std::string>>& target_prefix,
                      const TranslationOptions& options) override;

    private:
      std::vector<std::vector<std::vector<size_t>>>
      make_source_ids(const std::vector<std::vector<std::vector<std::string>>>& source_features,
                      size_t max_length = 0) const;

      std::vector<std::vector<size_t>>
      make_target_ids(const std::vector<std::vector<std::string>>& target,
                      size_t max_length = 0,
                      bool is_prefix = false) const;

      size_t get_source_length(const std::vector<std::string>& source,
                               bool include_special_tokens) const;

      void encode(const std::vector<std::vector<std::vector<size_t>>>& ids,
                  StorageView& memory,
                  StorageView& memory_lengths);

      const std::shared_ptr<const SequenceToSequenceModel> _model;
      const std::unique_ptr<layers::Encoder> _encoder;
      const std::unique_ptr<layers::Decoder> _decoder;
    };

  }
}
