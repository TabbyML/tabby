#pragma once

#include "ctranslate2/decoding.h"
#include "ctranslate2/layers/decoder.h"
#include "ctranslate2/layers/encoder.h"
#include "ctranslate2/models/model.h"
#include "ctranslate2/scoring.h"
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
                           const std::vector<std::vector<std::string>>& source,
                           StorageView& memory,
                           StorageView& memory_lengths) const;

      void forward_decoder(layers::Decoder& decoder,
                           layers::DecoderState& state,
                           const std::vector<std::vector<std::string>>& target,
                           StorageView& logits) const;

      void forward(layers::Encoder& encoder,
                   layers::Decoder& decoder,
                   const std::vector<std::vector<std::string>>& source,
                   const std::vector<std::vector<std::string>>& target,
                   StorageView& logits) const;

      std::vector<ScoringResult>
      score(layers::Encoder& encoder,
            layers::Decoder& decoder,
            const std::vector<std::vector<std::string>>& source,
            const std::vector<std::vector<std::string>>& target,
            const size_t max_input_length = 0) const;

      std::vector<GenerationResult<std::string>>
      sample(layers::Encoder& encoder,
             layers::Decoder& decoder,
             const std::vector<std::vector<std::string>>& source,
             const std::vector<std::vector<std::string>>& target_prefix = {},
             const SearchStrategy& search_strategy = GreedySearch(),
             const Sampler& sampler = BestSampler(),
             const bool use_vmap = false,
             const size_t max_input_length = 0,
             const size_t max_output_length = 256,
             const size_t min_output_length = 1,
             const size_t num_hypotheses = 1,
             const bool return_alternatives = false,
             const bool return_scores = false,
             const bool return_attention = false,
             const bool replace_unknowns = false,
             const bool normalize_scores = false,
             const float repetition_penalty = 1) const;

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
      std::shared_ptr<const Vocabulary> _source_vocabulary;
      std::shared_ptr<const Vocabulary> _target_vocabulary;
      std::unique_ptr<const VocabularyMap> _vocabulary_map;

      bool _with_source_bos = false;
      bool _with_source_eos = false;
      bool _with_target_bos = true;
    };

  }
}
