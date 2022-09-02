#include "ctranslate2/models/language_model.h"

#include "ctranslate2/decoding.h"

namespace ctranslate2 {
  namespace models {

    const Vocabulary& LanguageModel::get_vocabulary() const {
      return *_vocabulary;
    }

    void LanguageModel::initialize(ModelReader& model_reader) {
      VocabularyInfo vocab_info;
      vocab_info.unk_token = get_attribute_with_default<std::string>("unk_token", "<unk>");
      vocab_info.bos_token = get_attribute_with_default<std::string>("bos_token", "<s>");
      vocab_info.eos_token = get_attribute_with_default<std::string>("eos_token", "</s>");

      _vocabulary = std::make_unique<Vocabulary>(*model_reader.get_required_file("vocabulary.txt"),
                                                 std::move(vocab_info));
    }


    std::vector<ScoringResult>
    SequenceGeneratorReplica::score(const std::vector<std::vector<std::string>>& tokens,
                                    const ScoringOptions& options) {
      return get_batch_results_helper<ScoringResult>(
        tokens.size(),
        [this, &tokens, &options](size_t i, ScoringResult& result) {
          return skip_scoring(tokens[i], options, result);
        },
        [this, &tokens, &options](const std::vector<size_t>& index_to_run) {
          return run_scoring(index_vector(tokens, index_to_run), options);
        });
    }

    std::vector<GenerationResult>
    SequenceGeneratorReplica::generate(const std::vector<std::vector<std::string>>& start_tokens,
                                       const GenerationOptions& options) {
      if (start_tokens.empty())
        return {};
      return run_generation(start_tokens, options);
    }


    DecoderReplica::DecoderReplica(const std::shared_ptr<const LanguageModel>& model,
                                   std::unique_ptr<layers::Decoder> decoder)
      : SequenceGeneratorReplica(model)
      , _model(model)
      , _decoder(std::move(decoder))
    {
    }

    std::vector<ScoringResult>
    DecoderReplica::run_scoring(const std::vector<std::vector<std::string>>& tokens,
                                const ScoringOptions& options) {
      PROFILE("DecoderReplica::run_scoring");
      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const auto& vocabulary = _model->get_vocabulary();

      const auto ids = vocabulary.to_ids(tokens, options.max_input_length);

      layers::DecoderState state = _decoder->initial_state(/*iterative_decoding=*/false);
      return score_sequences(*_decoder,
                             state,
                             ids,
                             vocabulary,
                             _model->preferred_size_multiple());
    }

    bool DecoderReplica::skip_scoring(const std::vector<std::string>& tokens,
                                      const ScoringOptions&,
                                      ScoringResult&) {
      return tokens.size() < 2;
    }

    std::vector<GenerationResult>
    DecoderReplica::run_generation(const std::vector<std::vector<std::string>>& start_tokens,
                                   const GenerationOptions& options) {
      PROFILE("DecoderReplica::run_generation");
      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const auto& vocabulary = _model->get_vocabulary();
      _decoder->update_output_layer(_model->preferred_size_multiple());

      DecodingOptions decoding_options;
      decoding_options.beam_size = options.beam_size;
      decoding_options.length_penalty = options.length_penalty;
      decoding_options.repetition_penalty = options.repetition_penalty;
      decoding_options.no_repeat_ngram_size = options.no_repeat_ngram_size;
      decoding_options.disable_unk = options.disable_unk;
      decoding_options.allow_early_exit = options.allow_early_exit;
      decoding_options.max_length = options.max_length;
      decoding_options.min_length = options.min_length;
      decoding_options.sampling_topk = options.sampling_topk;
      decoding_options.sampling_temperature = options.sampling_temperature;
      decoding_options.num_hypotheses = options.num_hypotheses;
      decoding_options.normalize_scores = options.normalize_scores;
      decoding_options.return_scores = options.return_scores;
      decoding_options.return_alternatives = options.return_alternatives;
      decoding_options.min_alternative_expansion_prob = options.min_alternative_expansion_prob;

      const auto start_ids = vocabulary.to_ids(start_tokens);
      layers::DecoderState state = _decoder->initial_state();
      std::vector<DecodingResult> results = decode(*_decoder,
                                                   state,
                                                   start_ids,
                                                   vocabulary.eos_id(),
                                                   vocabulary.unk_id(),
                                                   decoding_options);

      std::vector<GenerationResult> final_results;
      final_results.reserve(results.size());
      for (size_t i = 0; i < results.size(); ++i) {
        auto& result = results[i];

        // Forward the start token to the output if it is not the special BOS token.
        if (!start_ids[i].empty() && start_ids[i][0] != vocabulary.bos_id()) {
          for (auto& sequence : result.hypotheses)
            sequence.insert(sequence.begin(), start_ids[i][0]);
        }

        final_results.emplace_back(vocabulary.to_tokens(result.hypotheses),
                                   std::move(result.scores));
      }

      return final_results;
    }

  }
}
