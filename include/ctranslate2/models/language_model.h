#pragma once

#include "ctranslate2/layers/decoder.h"
#include "ctranslate2/models/model.h"
#include "ctranslate2/generation.h"
#include "ctranslate2/scoring.h"
#include "ctranslate2/vocabulary.h"

namespace ctranslate2 {
  namespace models {

    // Base class for language models.
    class LanguageModel : public Model {
    public:
      const Vocabulary& get_vocabulary() const;

    protected:
      void initialize(ModelReader& model_reader) override;

    private:
      std::shared_ptr<const Vocabulary> _vocabulary;
    };


    // Base class for generative language models.
    class SequenceGeneratorReplica : public ModelReplica {
    public:
      SequenceGeneratorReplica(const std::shared_ptr<const LanguageModel>& model)
        : ModelReplica(model)
        , _model(model)
      {
      }

      static std::unique_ptr<SequenceGeneratorReplica> create_from_model(const Model& model) {
        return model.as_sequence_generator();
      }

      std::vector<ScoringResult>
      score(const std::vector<std::vector<std::string>>& tokens,
            const ScoringOptions& options = ScoringOptions());

      std::vector<GenerationResult>
      generate(const std::vector<std::vector<std::string>>& start_tokens,
               const GenerationOptions& options = GenerationOptions());

      StorageView forward(const std::vector<std::vector<std::string>>& tokens,
                          const bool return_log_probs);
      StorageView forward(const std::vector<std::vector<size_t>>& ids,
                          const bool return_log_probs);
      StorageView forward(const StorageView& ids,
                          const StorageView& lengths,
                          const bool return_log_probs);

    protected:
      virtual bool skip_scoring(const std::vector<std::string>& tokens,
                                const ScoringOptions& options,
                                ScoringResult& result) {
        (void)tokens;
        (void)options;
        (void)result;
        return false;
      }

      virtual std::vector<ScoringResult>
      run_scoring(const std::vector<std::vector<std::string>>& tokens,
                  const ScoringOptions& options) = 0;

      virtual std::vector<GenerationResult>
      run_generation(const std::vector<std::vector<std::string>>& start_tokens,
                     const GenerationOptions& options) = 0;

      virtual StorageView forward(const StorageView& ids, const StorageView& lengths) = 0;

    private:
      const std::shared_ptr<const LanguageModel> _model;
    };


    // A model generating sequences with a decoder.
    class DecoderReplica : public SequenceGeneratorReplica {
    public:
      DecoderReplica(const std::shared_ptr<const LanguageModel>& model,
                     std::unique_ptr<layers::Decoder> decoder);

    protected:
      bool skip_scoring(const std::vector<std::string>& tokens,
                        const ScoringOptions& options,
                        ScoringResult& result) override;

      std::vector<ScoringResult>
      run_scoring(const std::vector<std::vector<std::string>>& tokens,
                  const ScoringOptions& options) override;

      std::vector<GenerationResult>
      run_generation(const std::vector<std::vector<std::string>>& start_tokens,
                     const GenerationOptions& options) override;

      StorageView forward(const StorageView& ids, const StorageView& lengths);

    private:
      const std::shared_ptr<const LanguageModel> _model;
      const std::unique_ptr<layers::Decoder> _decoder;
    };

  }
}
