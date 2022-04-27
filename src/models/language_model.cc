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


    DecoderReplica::DecoderReplica(const std::shared_ptr<const LanguageModel>& model,
                                   std::unique_ptr<layers::Decoder> decoder)
      : SequenceGeneratorReplica(model)
      , _model(model)
      , _decoder(std::move(decoder))
    {
    }

    std::vector<ScoringResult>
    DecoderReplica::score(const std::vector<std::vector<std::string>>& tokens,
                          const ScoringOptions& options) {
      PROFILE("DecoderReplica::score");
      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const auto& vocabulary = _model->get_vocabulary();

      std::vector<std::vector<size_t>> ids = vocabulary.to_ids(tokens);
      if (options.max_input_length > 0)
        truncate_sequences(ids, options.max_input_length);

      layers::DecoderState state = _decoder->initial_state(/*iterative_decoding=*/false);
      return score_sequences(*_decoder,
                             state,
                             ids,
                             vocabulary,
                             _model->preferred_size_multiple());
    }

    std::vector<GenerationResult>
    DecoderReplica::generate(const std::vector<std::vector<std::string>>& start_tokens,
                             const GenerationOptions& options) {
      PROFILE("DecoderReplica::generate");
      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const auto& vocabulary = _model->get_vocabulary();

      std::vector<size_t> include_ids;
      std::vector<size_t> exclude_ids;
      if (options.disable_unk)
        exclude_ids = {vocabulary.unk_id()};
      const auto* output_ids_map = _decoder->update_output_layer(_model->preferred_size_multiple(),
                                                                 include_ids,
                                                                 exclude_ids);

      DecodingOptions decoding_options;
      decoding_options.beam_size = options.beam_size;
      decoding_options.length_penalty = options.length_penalty;
      decoding_options.repetition_penalty = options.repetition_penalty;
      decoding_options.allow_early_exit = options.allow_early_exit;
      decoding_options.max_length = options.max_length;
      decoding_options.min_length = options.min_length;
      decoding_options.sampling_topk = options.sampling_topk;
      decoding_options.sampling_temperature = options.sampling_temperature;
      decoding_options.num_hypotheses = options.num_hypotheses;
      decoding_options.normalize_scores = options.normalize_scores;
      decoding_options.return_scores = options.return_scores;
      decoding_options.return_alternatives = options.return_alternatives;

      layers::DecoderState state = _decoder->initial_state();
      std::vector<DecodingResult> results = decode(*_decoder,
                                                   state,
                                                   vocabulary.to_ids(start_tokens),
                                                   vocabulary.eos_id(),
                                                   decoding_options,
                                                   output_ids_map);

      std::vector<GenerationResult> final_results;
      final_results.reserve(results.size());
      for (auto& result : results) {
        final_results.emplace_back(vocabulary.to_tokens(result.hypotheses),
                                   std::move(result.scores));
      }

      return final_results;
    }

  }
}
