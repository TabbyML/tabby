#include "ctranslate2/models/language_model.h"

#include "ctranslate2/decoding.h"

namespace ctranslate2 {
  namespace models {

    const Vocabulary& LanguageModel::get_vocabulary() const {
      return *_vocabulary;
    }

    void LanguageModel::initialize(ModelReader& model_reader) {
      if (binary_version() < 6) {
        config["unk_token"] = get_attribute_with_default<std::string>("unk_token", "<unk>");
        config["bos_token"] = get_attribute_with_default<std::string>("bos_token", "<s>");
        config["eos_token"] = get_attribute_with_default<std::string>("eos_token", "</s>");
      }

      VocabularyInfo vocab_info;
      vocab_info.unk_token = config["unk_token"];
      vocab_info.bos_token = config["bos_token"];
      vocab_info.eos_token = config["eos_token"];

      _vocabulary = std::make_shared<Vocabulary>(*model_reader.get_required_file("vocabulary.txt"),
                                                 std::move(vocab_info));
    }


    std::vector<ScoringResult>
    SequenceGeneratorReplica::score(const std::vector<std::vector<std::string>>& tokens,
                                    const ScoringOptions& options) {
      PROFILE("SequenceGeneratorReplica::score");
      const auto scoped_device_setter = model()->get_scoped_device_setter();

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
      PROFILE("SequenceGeneratorReplica::generate");
      const auto scoped_device_setter = model()->get_scoped_device_setter();

      if (start_tokens.empty())
        return {};
      return run_generation(start_tokens, options);
    }

    StorageView
    SequenceGeneratorReplica::forward(const std::vector<std::vector<std::string>>& tokens,
                                      const bool return_log_probs) {
      const auto& vocabulary = _model->get_vocabulary();
      return forward(vocabulary.to_ids(tokens), return_log_probs);
    }

    StorageView
    SequenceGeneratorReplica::forward(const std::vector<std::vector<size_t>>& ids,
                                      const bool return_log_probs) {
      StorageView lengths;
      StorageView input_ids = layers::make_sequence_inputs(ids, Device::CPU, 1, &lengths);
      return forward(input_ids, lengths, return_log_probs);
    }

    StorageView
    SequenceGeneratorReplica::forward(const StorageView& ids,
                                      const StorageView& lengths,
                                      const bool return_log_probs) {
      PROFILE("SequenceGeneratorReplica::forward");
      const auto& model = *this->model();
      const auto device = model.device();
      const auto scoped_device_setter = model.get_scoped_device_setter();

      StorageView output;
      if (ids.device() != device)
        output = forward(ids.to(device), lengths.to(device));
      else
        output = forward(ids, lengths);

      if (return_log_probs)
        ops::LogSoftMax()(output);

      // Ensure all operations are finished before returning the output.
      synchronize_stream(model.device());
      return output;
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
      const auto& vocabulary = _model->get_vocabulary();
      _decoder->update_output_layer(_model->preferred_size_multiple());

      DecodingOptions decoding_options;
      decoding_options.beam_size = options.beam_size;
      decoding_options.patience = options.patience;
      decoding_options.length_penalty = options.length_penalty;
      decoding_options.repetition_penalty = options.repetition_penalty;
      decoding_options.no_repeat_ngram_size = options.no_repeat_ngram_size;
      decoding_options.max_length = options.max_length;
      decoding_options.min_length = options.min_length;
      decoding_options.sampling_topk = options.sampling_topk;
      decoding_options.sampling_temperature = options.sampling_temperature;
      decoding_options.num_hypotheses = options.num_hypotheses;
      decoding_options.return_scores = options.return_scores;
      decoding_options.return_alternatives = options.return_alternatives;
      decoding_options.min_alternative_expansion_prob = options.min_alternative_expansion_prob;
      decoding_options.disable_sequences = vocabulary.to_ids(options.suppress_sequences);
      if (options.disable_unk)
        decoding_options.disable_ids.push_back(vocabulary.unk_id());


      const auto start_ids = vocabulary.to_ids(start_tokens);
      const auto end_id = (options.end_token.empty()
                           ? vocabulary.eos_id()
                           : vocabulary.to_id(options.end_token));
      layers::DecoderState state = _decoder->initial_state();
      std::vector<DecodingResult> results = decode(*_decoder,
                                                   state,
                                                   start_ids,
                                                   end_id,
                                                   decoding_options);

      std::vector<GenerationResult> final_results;
      final_results.reserve(results.size());
      for (size_t i = 0; i < results.size(); ++i) {
        auto& result = results[i];

        // Remove EOS token.
        for (auto& sequence : result.hypotheses) {
          while (!sequence.empty() && sequence.back() == end_id)
            sequence.pop_back();
        }

        // Forward the start token to the output if it is not the special BOS token.
        if (!start_ids[i].empty() && start_ids[i][0] != vocabulary.bos_id()) {
          for (auto& sequence : result.hypotheses)
            sequence.insert(sequence.begin(), start_ids[i][0]);
        }

        GenerationResult final_result;
        final_result.sequences = vocabulary.to_tokens(result.hypotheses);
        final_result.sequences_ids = std::move(result.hypotheses);
        final_result.scores = std::move(result.scores);
        final_results.emplace_back(std::move(final_result));
      }

      return final_results;
    }

    StorageView DecoderReplica::forward(const StorageView& ids, const StorageView& lengths) {
      if (ids.rank() != 2)
        throw std::invalid_argument("Expected input ids to have 2 dimensions, but got "
                                    + std::to_string(ids.rank())
                                    + " dimension(s) instead");
      if (lengths.size() != ids.dim(0))
        throw std::invalid_argument("Expected lengths vector to have size "
                                    + std::to_string(ids.dim(0))
                                    + ", but got size "
                                    + std::to_string(lengths.size())
                                    + " instead");

      auto& decoder = *_decoder;

      decoder.update_output_layer(_model->preferred_size_multiple());
      auto state = decoder.initial_state(/*iterative_decoding=*/false);

      StorageView logits(decoder.output_type(), decoder.device());
      decoder(ids, lengths, state, logits);
      return logits;
    }

  }
}
