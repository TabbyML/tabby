#include "ctranslate2/models/language_model.h"

#include "ctranslate2/decoding.h"

namespace ctranslate2 {
  namespace models {

    LanguageModel::LanguageModel()
      : _state_cache(std::make_shared<layers::DecoderStateCache>())
    {
    }

    const Vocabulary& LanguageModel::get_vocabulary() const {
      return *_vocabulary;
    }

    layers::DecoderStateCache& LanguageModel::get_state_cache() const {
      return *_state_cache;
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

      _vocabulary = load_vocabulary(model_reader, "vocabulary", std::move(vocab_info));
      if (!_vocabulary)
        throw std::runtime_error("Cannot load the vocabulary from the model directory");
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
                             _model->preferred_size_multiple(),
                             options.offset);
    }

    bool DecoderReplica::skip_scoring(const std::vector<std::string>& tokens,
                                      const ScoringOptions&,
                                      ScoringResult&) {
      return tokens.size() < 2;
    }

    static void copy_state(const layers::DecoderState& from,
                           layers::DecoderState& to,
                           dim_t batch_size) {
      if (batch_size == 1) {
        for (const auto& [name, value] : from)
          to[name] = value;

      } else {
        const ops::Tile tile_op(/*axis=*/0, /*repeats=*/batch_size);
        for (const auto& [name, value] : from)
          tile_op(value, to[name]);
      }
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
      decoding_options.sampling_topp = options.sampling_topp;
      decoding_options.sampling_temperature = options.sampling_temperature;
      decoding_options.num_hypotheses = options.num_hypotheses;
      decoding_options.return_scores = options.return_scores;
      decoding_options.return_alternatives = options.return_alternatives;
      decoding_options.min_alternative_expansion_prob = options.min_alternative_expansion_prob;
      decoding_options.disable_sequences = vocabulary.to_ids(options.suppress_sequences,
                                                             /*max_length=*/0,
                                                             /*prefix=*/nullptr,
                                                             /*suffix=*/nullptr,
                                                             /*allow_unk=*/false);

      if (options.disable_unk)
        decoding_options.disable_ids.push_back(vocabulary.unk_id());
      if (options.callback)
        decoding_options.callback = [&options, &vocabulary](DecodingStepResult step_result) -> bool {
          return options.callback(GenerationStepResult(step_result, vocabulary));
        };

      std::vector<std::vector<size_t>> start_ids = vocabulary.to_ids(start_tokens);
      layers::DecoderState state = _decoder->initial_state();

      if (!options.static_prompt.empty()) {
        std::vector<size_t> static_prompt_ids;
        static_prompt_ids.reserve(options.static_prompt.size());
        for (const auto& token : options.static_prompt)
          static_prompt_ids.emplace_back(vocabulary.to_id(token));

        auto& cache = _model->get_state_cache();
        const dim_t batch_size = start_ids.size();
        const layers::DecoderState* cached_state = (options.cache_static_prompt
                                                    ? cache.get(static_prompt_ids)
                                                    : nullptr);

        if (cached_state) {
          copy_state(*cached_state, state, batch_size);

        } else {
          layers::DecoderState static_state = _decoder->initial_state();
          StorageView static_prompt = layers::make_sequence_inputs({static_prompt_ids},
                                                                   _decoder->device());

          (*_decoder)(0, static_prompt, static_state);
          copy_state(static_state, state, batch_size);

          if (options.cache_static_prompt)
            cache.save(static_prompt_ids, std::move(static_state));
        }

        decoding_options.start_step += static_prompt_ids.size();
      }

      if (!options.include_prompt_in_result) {
        size_t min_prompt_length = start_ids[0].size();
        for (const auto& start_sequence : start_ids)
          min_prompt_length = std::min(min_prompt_length, start_sequence.size());

        size_t forward_length = min_prompt_length - 1;

        if (forward_length > 0) {
          std::vector<std::vector<size_t>> prompt_ids;
          prompt_ids.reserve(start_ids.size());
          for (auto& start_sequence : start_ids) {
            prompt_ids.emplace_back(start_sequence.begin(), start_sequence.begin() + forward_length);
            start_sequence.erase(start_sequence.begin(), start_sequence.begin() + forward_length);
          }

          StorageView prompt = layers::make_sequence_inputs(prompt_ids, _decoder->device());
          (*_decoder)(decoding_options.start_step, prompt, state);

          decoding_options.start_step += prompt.dim(1);
          decoding_options.return_prefix = false;
        }
      }

      const auto end_ids(std::visit(ResolveEndToken(vocabulary), options.end_token));
      std::vector<DecodingResult> results = decode(*_decoder,
                                                   state,
                                                   start_ids,
                                                   end_ids,
                                                   decoding_options);

      std::vector<GenerationResult> final_results;
      final_results.reserve(results.size());
      for (size_t i = 0; i < results.size(); ++i) {
        auto& result = results[i];

        // Remove EOS token.
        if (!options.return_end_token) {
          for (auto& sequence : result.hypotheses) {
            while (!sequence.empty() && is_eos(sequence.back(), end_ids))
              sequence.pop_back();
          }
        }

        // Forward the start token to the output if it is not the special BOS token.
        if (options.include_prompt_in_result
            && !start_ids[i].empty()
            && start_ids[i][0] != vocabulary.bos_id()) {
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


    EncoderForwardOutput
    SequenceEncoderReplica::forward(const std::vector<std::vector<std::string>>& tokens,
                                    const std::vector<std::vector<size_t>>& token_type_ids) {
      const auto& vocabulary = _model->get_vocabulary();
      return forward(vocabulary.to_ids(tokens), token_type_ids);
    }

    EncoderForwardOutput
    SequenceEncoderReplica::forward(const std::vector<std::vector<size_t>>& ids,
                                    const std::vector<std::vector<size_t>>& token_type_ids) {
      StorageView lengths;
      StorageView input_ids = layers::make_sequence_inputs(ids, Device::CPU, 1, &lengths);
      return forward(input_ids, lengths, token_type_ids);
    }

    EncoderForwardOutput
    SequenceEncoderReplica::forward(const StorageView& ids,
                                    const StorageView& lengths,
                                    const std::vector<std::vector<size_t>>& token_type_ids) {
      PROFILE("SequenceEncoderReplica::forward");
      const auto& model = *this->model();
      const auto device = model.device();
      const auto scoped_device_setter = model.get_scoped_device_setter();

      StorageView input_token_type_ids = layers::make_sequence_inputs(token_type_ids, device);
      EncoderForwardOutput output;

      if (ids.device() != device)
        output = forward_impl(ids.to(device), lengths.to(device), input_token_type_ids);
      else
        output = forward_impl(ids, lengths, input_token_type_ids);

      // Ensure all operations are finished before returning the output.
      synchronize_stream(device);
      return output;
    }


    EncoderReplica::EncoderReplica(const std::shared_ptr<const LanguageModel>& model,
                                   std::unique_ptr<layers::Encoder> encoder)
      : SequenceEncoderReplica(model)
      , _model(model)
      , _encoder(std::move(encoder))
      , _pooler_activation(model->get_enum_value<ops::ActivationType>("pooler_activation"))
      , _pooler_dense(layers::build_optional_layer<layers::Dense>(*model,
                                                                  "pooler_dense",
                                                                  &_pooler_activation))
    {
    }

    EncoderForwardOutput
    EncoderReplica::forward_impl(const StorageView& ids,
                                 const StorageView& lengths,
                                 const StorageView& token_type_ids) {
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

      const Device device = _model->device();
      const DataType dtype = _encoder->output_type();

      std::vector<StorageView> inputs{ids};

      if (_encoder->num_input_features() > 1) {
        if (token_type_ids.empty()) {
          StorageView placeholder_type_ids(ids.shape(), ids.dtype(), device);
          placeholder_type_ids.zero();
          inputs.emplace_back(std::move(placeholder_type_ids));
        } else {
          inputs.emplace_back(token_type_ids);
        }
      }

      StorageView last_hidden_state(dtype, device);
      (*_encoder)(inputs, lengths, last_hidden_state);

      EncoderForwardOutput output;
      output.last_hidden_state = std::move(last_hidden_state);

      if (_pooler_dense) {
        StorageView first_index({ids.dim(0)}, int32_t(0), device);
        StorageView first_token_state(dtype, device);

        ops::Gather(/*axis=*/1, /*batch_dims=*/1)(output.last_hidden_state,
                                                  first_index,
                                                  first_token_state);

        StorageView pooler_output(dtype, device);
        (*_pooler_dense)(first_token_state, pooler_output);

        output.pooler_output = std::move(pooler_output);
      }

      return output;
    }

  }
}
