#include "ctranslate2/models/whisper.h"

#include <algorithm>

#include "ctranslate2/decoding.h"
#include "ctranslate2/models/model_factory.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace models {

    static auto register_whisper = register_model<WhisperModel>("WhisperSpec");

    const Vocabulary& WhisperModel::get_vocabulary() const {
      return *_vocabulary;
    }

    size_t WhisperModel::current_spec_revision() const {
      return 3;
    }

    void WhisperModel::initialize(ModelReader& model_reader) {
      VocabularyInfo vocab_info;
      vocab_info.unk_token = "<|endoftext|>";
      vocab_info.bos_token = "<|startoftranscript|>";
      vocab_info.eos_token = "<|endoftext|>";
      _vocabulary = std::make_shared<Vocabulary>(*model_reader.get_required_file("vocabulary.txt"),
                                                 std::move(vocab_info));
    }

    bool WhisperModel::is_quantizable(const std::string& variable_name) const {
      return (Model::is_quantizable(variable_name)
              && variable_name.find("conv") == std::string::npos);
    }

    bool WhisperModel::is_linear_weight(const std::string& variable_name) const {
      return is_quantizable(variable_name) && variable_name.find("embeddings") == std::string::npos;
    }

    std::unique_ptr<Model> WhisperModel::clone() const {
      return std::make_unique<WhisperModel>(*this);
    }


    std::unique_ptr<WhisperReplica> WhisperReplica::create_from_model(const Model& model) {
      if (!dynamic_cast<const WhisperModel*>(&model))
        throw std::invalid_argument("The model is not a Whisper model");

      const auto scoped_device_setter = model.get_scoped_device_setter();
      const auto model_ptr = model.shared_from_this();
      const auto concrete_model = std::static_pointer_cast<const WhisperModel>(model_ptr);
      return std::make_unique<WhisperReplica>(concrete_model);
    }

    WhisperReplica::WhisperReplica(const std::shared_ptr<const WhisperModel>& model)
      : ModelReplica(model)
      , _model(model)
      , _encoder(std::make_unique<layers::WhisperEncoder>(*model, "encoder"))
      , _decoder(std::make_unique<layers::WhisperDecoder>(*model, "decoder"))
    {
    }

    StorageView WhisperReplica::encode(const StorageView& features) {
      const Device device = _model->device();
      const DataType dtype = _encoder->output_type();

      StorageView encoder_output(dtype, device);
      if (features.device() == device && features.dtype() == dtype)
        (*_encoder)(features, encoder_output);
      else
        (*_encoder)(features.to(device).to(dtype), encoder_output);

      return encoder_output;
    }

    std::vector<WhisperGenerationResult>
    WhisperReplica::generate(const StorageView& features,
                             const std::vector<std::vector<std::string>>& prompts,
                             const WhisperOptions& options) {
      const auto& vocabulary = _model->get_vocabulary();
      return generate(features, vocabulary.to_ids(prompts), options);
    }

    class ApplyTimestampRules;

    static std::shared_ptr<ApplyTimestampRules>
    create_timestamp_processor(const std::vector<std::vector<size_t>>& prompts,
                               const Vocabulary& vocabulary) {
      const size_t no_timestamps_id = vocabulary.to_id("<|notimestamps|>");

      bool any_with_timestamps = false;
      std::vector<bool> with_timestamps;
      with_timestamps.reserve(prompts.size());

      for (const auto& prompt : prompts) {
        const auto no_ts_it = std::find(prompt.begin(), prompt.end(), no_timestamps_id);
        const auto generate_timestamps = (no_ts_it == prompt.end());
        with_timestamps.emplace_back(generate_timestamps);
        if (generate_timestamps)
          any_with_timestamps = true;
      }

      if (!any_with_timestamps)
        return nullptr;

      const size_t eot_id = vocabulary.eos_id();
      const size_t timestamp_begin_id = no_timestamps_id + 1;
      const size_t timestamp_end_id = vocabulary.size() - 1;
      const size_t max_initial_timestamp_id = timestamp_begin_id + 50;

      return std::make_shared<ApplyTimestampRules>(eot_id,
                                                   no_timestamps_id,
                                                   timestamp_begin_id,
                                                   timestamp_end_id,
                                                   max_initial_timestamp_id,
                                                   std::move(with_timestamps));
    }

    static StorageView get_sot_index(const std::vector<std::vector<size_t>>& prompts,
                                     const size_t sot_id,
                                     const Device device) {
      const dim_t batch_size = prompts.size();
      StorageView sot_index({batch_size}, DataType::INT32);

      for (dim_t i = 0; i < batch_size; ++i) {
        const auto& prompt = prompts[i];
        const auto sot_it = std::find(prompt.begin(), prompt.end(), sot_id);
        if (sot_it == prompt.end())
          throw std::invalid_argument("<|startoftranscript|> token was not found in the "
                                      "prompt of batch " + std::to_string(i));

        sot_index.at<int32_t>(i) = std::distance(prompt.begin(), sot_it);
      }

      return sot_index.to(device);
    }

    static std::vector<float> get_no_speech_probs(layers::WhisperDecoder& decoder,
                                                  const StorageView& decoder_outputs,
                                                  const StorageView& sot_index,
                                                  const size_t no_speech_id) {
      const Device device = decoder.device();
      const DataType dtype = decoder.output_type();

      StorageView logits(dtype, device);
      decoder.compute_logits_for_steps(decoder_outputs, sot_index, logits);

      ops::SoftMax()(logits);
      StorageView probs_at_sot = std::move(logits);

      StorageView gather_ids({probs_at_sot.dim(0)}, int32_t(no_speech_id), device);
      StorageView no_speech_probs(dtype, device);
      ops::Gather(/*axis=*/1, /*batch_dims=*/1)(probs_at_sot, gather_ids, no_speech_probs);

      if (no_speech_probs.dtype() != DataType::FLOAT)
        no_speech_probs = no_speech_probs.to_float();
      return no_speech_probs.to_vector<float>();
    }

    std::vector<WhisperGenerationResult>
    WhisperReplica::generate(const StorageView& features,
                             const std::vector<std::vector<size_t>>& prompts,
                             const WhisperOptions& options) {
      PROFILE("WhisperReplica::generate");
      if (prompts.empty())
        return {};

      const bool fixed_length = std::all_of(prompts.begin(),
                                            prompts.end(),
                                            [&prompts](const std::vector<size_t>& prompt) {
                                              return prompt.size() == prompts[0].size();
                                            });

      if (!fixed_length)
        throw std::invalid_argument("The generate method currently requires all prompts to "
                                    "have the same length. To work around this limitation, "
                                    "simply adapt the number of previous text tokens in each "
                                    "batch.");

      if (prompts[0].size() < 3)
        throw std::invalid_argument("The prompt should have at least 3 tokens: "
                                    "START OF TRANSCRIPT, LANGUAGE TAG, and TRANSCRIBE/TRANSLATE");

      const auto& vocabulary = _model->get_vocabulary();
      const auto scoped_device_setter = _model->get_scoped_device_setter();

      layers::DecoderState state = _decoder->initial_state();
      state.emplace("memory", encode(features));

      _decoder->update_output_layer(_model->preferred_size_multiple());

      std::vector<std::vector<size_t>> prefix_tokens;
      std::vector<std::vector<size_t>> start_tokens;
      prefix_tokens.reserve(prompts.size());
      start_tokens.reserve(prompts.size());
      for (const auto& prompt : prompts) {
        prefix_tokens.emplace_back(prompt.begin(), prompt.end() - 1);
        start_tokens.emplace_back(prompt.end() - 1, prompt.end());
      }

      const Device device = _decoder->device();
      const DataType dtype = _decoder->output_type();
      StorageView inputs = layers::make_sequence_inputs(prefix_tokens, device);
      StorageView outputs(dtype, device);

      // Initialize the decoder state with the prompt.
      _decoder->forward_prompt(inputs, state, options.return_no_speech_prob ? &outputs : nullptr);

      std::vector<float> no_speech_probs;
      if (options.return_no_speech_prob) {
        // Get the probability of the no speech token at the start of transcript step.
        StorageView sot_index = get_sot_index(prefix_tokens, vocabulary.bos_id(), device);
        size_t no_speech_id = vocabulary.to_id("<|nospeech|>");
        if (no_speech_id == vocabulary.unk_id())
          no_speech_id = vocabulary.to_id("<|nocaptions|>");
        no_speech_probs = get_no_speech_probs(*_decoder, outputs, sot_index, no_speech_id);
      }

      DecodingOptions decoding_options;
      decoding_options.start_step = inputs.dim(1);
      decoding_options.beam_size = options.beam_size;
      decoding_options.patience = options.patience;
      decoding_options.length_penalty = options.length_penalty;
      decoding_options.repetition_penalty = options.repetition_penalty;
      decoding_options.no_repeat_ngram_size = options.no_repeat_ngram_size;
      decoding_options.max_length = options.max_length / 2;
      decoding_options.sampling_topk = options.sampling_topk;
      decoding_options.sampling_temperature = options.sampling_temperature;
      decoding_options.num_hypotheses = options.num_hypotheses;
      decoding_options.return_scores = options.return_scores;
      decoding_options.return_prefix = false;
      for (const auto& id : _model->config["suppress_ids"])
        decoding_options.disable_ids.push_back(id);
      for (const auto& id : _model->config["suppress_ids_begin"])
        decoding_options.disable_ids_begin.push_back(id);

      const auto timestamp_processor = create_timestamp_processor(prompts, vocabulary);
      if (timestamp_processor)
        decoding_options.logits_processors.emplace_back(timestamp_processor);

      std::vector<DecodingResult> results = decode(*_decoder,
                                                   state,
                                                   start_tokens,
                                                   vocabulary.eos_id(),
                                                   decoding_options);

      std::vector<WhisperGenerationResult> final_results;
      final_results.reserve(results.size());

      for (size_t i = 0; i < results.size(); ++i) {
        auto& result = results[i];

        WhisperGenerationResult final_result;
        final_result.sequences = vocabulary.to_tokens(result.hypotheses);
        final_result.sequences_ids = std::move(result.hypotheses);
        final_result.scores = std::move(result.scores);
        if (options.return_no_speech_prob)
          final_result.no_speech_prob = no_speech_probs[i];

        final_results.emplace_back(std::move(final_result));
      }

      return final_results;
    }

    std::vector<std::vector<std::pair<std::string, float>>>
    WhisperReplica::detect_language(const StorageView& features) {
      PROFILE("WhisperReplica::detect_language");

      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const auto& vocabulary = _model->get_vocabulary();
      const auto device = _model->device();

      const int32_t sot = vocabulary.bos_id();
      std::vector<int32_t> lang_ids;
      for (const auto& id : _model->config["lang_ids"])
        lang_ids.push_back(id);

      const dim_t batch_size = features.dim(0);
      const dim_t num_langs = lang_ids.size();

      StorageView start_ids({batch_size}, sot, device);
      StorageView score_ids({batch_size, num_langs}, DataType::INT32);
      for (dim_t i = 0; i < batch_size; ++i) {
        for (dim_t j = 0; j < num_langs; ++j)
          score_ids.at<int32_t>({i, j}) = lang_ids[j];
      }
      if (score_ids.device() != device)
        score_ids = score_ids.to(device);

      layers::DecoderState state = _decoder->initial_state();
      state.emplace("memory", encode(features));

      StorageView logits(_decoder->output_type(), device);
      StorageView lang_probs(logits.dtype(), device);
      (*_decoder)(0, start_ids, state, &logits);
      ops::Gather(/*axis=*/-1, /*batch_dims=*/1)(logits, score_ids, lang_probs);
      ops::SoftMax()(lang_probs);

      if (lang_probs.dtype() != DataType::FLOAT)
        lang_probs = lang_probs.to_float();
      if (lang_probs.device() != Device::CPU)
        lang_probs = lang_probs.to(Device::CPU);

      std::vector<std::vector<std::pair<std::string, float>>> results;
      results.reserve(batch_size);

      for (dim_t i = 0; i < batch_size; ++i) {
        std::vector<std::pair<std::string, float>> result;
        result.reserve(num_langs);

        for (dim_t j = 0; j < num_langs; ++j) {
          const size_t lang_id = lang_ids[j];
          const float prob = lang_probs.at<float>({i, j});
          result.emplace_back(vocabulary.to_token(lang_id), prob);
        }

        std::sort(result.begin(), result.end(),
                  [](const std::pair<std::string, float>& a,
                     const std::pair<std::string, float>& b) {
                    return a.second > b.second;
                  });

        results.emplace_back(std::move(result));
      }

      return results;
    }


    std::vector<std::future<WhisperGenerationResult>>
    Whisper::generate(StorageView features,
                      std::vector<std::vector<std::string>> prompts,
                      WhisperOptions options) {
      const size_t batch_size = features.dim(0);
      return post_batch<WhisperGenerationResult>(
        [features = std::move(features), prompts = std::move(prompts), options]
        (WhisperReplica& replica) {
          return replica.generate(features, prompts, options);
        },
        batch_size);
    }

    std::vector<std::future<WhisperGenerationResult>>
    Whisper::generate(StorageView features,
                      std::vector<std::vector<size_t>> prompts,
                      WhisperOptions options) {
      const size_t batch_size = features.dim(0);
      return post_batch<WhisperGenerationResult>(
        [features = std::move(features), prompts = std::move(prompts), options]
        (WhisperReplica& replica) {
          return replica.generate(features, prompts, options);
        },
        batch_size);
    }

    std::vector<std::future<std::vector<std::pair<std::string, float>>>>
    Whisper::detect_language(StorageView features) {
      const size_t batch_size = features.dim(0);
      return post_batch<std::vector<std::pair<std::string, float>>>(
        [features = std::move(features)](WhisperReplica& replica) {
          return replica.detect_language(features);
        },
        batch_size);
    }


    class ApplyTimestampRules : public LogitsProcessor {
    private:
      const size_t _eot_id;
      const size_t _no_timestamps_id;
      const size_t _timestamp_begin_id;
      const size_t _timestamp_end_id;
      const size_t _max_initial_timestamp_id;
      const std::vector<bool> _batch_with_timestamps;

    public:
      ApplyTimestampRules(const size_t eot_id,
                          const size_t no_timestamps_id,
                          const size_t timestamp_begin_id,
                          const size_t timestamp_end_id,
                          const size_t max_initial_timestamp_id,
                          std::vector<bool> batch_with_timestamps)
        : _eot_id(eot_id)
        , _no_timestamps_id(no_timestamps_id)
        , _timestamp_begin_id(timestamp_begin_id)
        , _timestamp_end_id(timestamp_end_id)
        , _max_initial_timestamp_id(max_initial_timestamp_id)
        , _batch_with_timestamps(std::move(batch_with_timestamps))
      {
      }

      void apply(dim_t step,
                 StorageView& logits,
                 DisableTokens& disable_tokens,
                 const StorageView& sequences,
                 const std::vector<dim_t>& batch_offset,
                 const std::vector<std::vector<size_t>>* prefix) override {
        std::vector<dim_t> check_timestamps_prob_for_batch;
        const dim_t batch_size = logits.dim(0);

        for (dim_t batch_id = 0; batch_id < batch_size; ++batch_id) {
          if (!_batch_with_timestamps[get_batch_index(batch_size, batch_id, batch_offset)])
            continue;

          const dim_t sample_begin = get_sample_begin(batch_size, batch_id, batch_offset, prefix);

          // Suppress <|notimestamps|>.
          disable_tokens.add(batch_id, _no_timestamps_id);

          if (step == sample_begin) {
            // Suppress non timestamps at the beginning.
            for (size_t i = 0; i < _timestamp_begin_id; ++i)
              disable_tokens.add(batch_id, i);

            // Apply max_initial_timestamp option.
            for (size_t i = _max_initial_timestamp_id + 1; i <= _timestamp_end_id; ++i)
              disable_tokens.add(batch_id, i);

          } else if (step > sample_begin) {
            // Timestamps have to appear in pairs, except directly before EOT.
            const size_t last_token = sequences.at<int32_t>({batch_id, step - 1});

            if (last_token >= _timestamp_begin_id) {
              const size_t penultimate_token = (step - 1 > sample_begin
                                                ? sequences.at<int32_t>({batch_id, step - 2})
                                                : last_token);

              if (penultimate_token >= _timestamp_begin_id) {  // has to be non-timestamp
                for (size_t i = _timestamp_begin_id; i <= _timestamp_end_id; ++i)
                  disable_tokens.add(batch_id, i);
              } else {  // cannot be normal text tokens
                for (size_t i = 0; i < _eot_id; ++i)
                  disable_tokens.add(batch_id, i);
              }
            } else {
              check_timestamps_prob_for_batch.push_back(batch_id);
            }

            // Timestamps shouldn't decrease: forbid timestamp tokens smaller than the last.
            for (dim_t t = step - 1; t >= sample_begin; --t) {
              const size_t token = sequences.at<int32_t>({batch_id, t});

              if (token >= _timestamp_begin_id) {
                for (size_t i = _timestamp_begin_id; i <= token; ++i)
                  disable_tokens.add(batch_id, i);
                break;
              }
            }
          }
        }

        if (!check_timestamps_prob_for_batch.empty()) {
          // Apply all changes to the logits before computing the log softmax.
          disable_tokens.apply();

          StorageView log_probs(logits.dtype(), logits.device());
          ops::LogSoftMax()(logits, log_probs);

          for (const dim_t batch_id : check_timestamps_prob_for_batch) {
            bool sample_timestamp = false;

            if (log_probs.device() == Device::CPU)
              sample_timestamp = should_sample_timestamp<Device::CPU, float>(log_probs, batch_id);
#ifdef CT2_WITH_CUDA
            else if (log_probs.dtype() == DataType::FLOAT)
              sample_timestamp = should_sample_timestamp<Device::CUDA, float>(log_probs, batch_id);
            else
              sample_timestamp = should_sample_timestamp<Device::CUDA, float16_t>(log_probs, batch_id);
#endif

            if (sample_timestamp) {
              for (size_t i = 0; i < _timestamp_begin_id; ++i)
                disable_tokens.add(batch_id, i);
            }
          }
        }
      }

      template <Device D, typename T>
      bool should_sample_timestamp(const StorageView& log_probs, const dim_t batch_id) {
        const dim_t num_text_tokens = _timestamp_begin_id;
        const dim_t num_timestamp_tokens = _timestamp_end_id - _timestamp_begin_id + 1;

        const T* text_log_probs = log_probs.index<T>({batch_id, 0});
        const T* timestamp_log_probs = text_log_probs + num_text_tokens;

        // If sum of probability over timestamps is above any other token, sample timestamp.
        const float max_text_token_log_prob = primitives<D>::max(text_log_probs, num_text_tokens);
        const float timestamp_log_prob = primitives<D>::logsumexp(timestamp_log_probs,
                                                                  num_timestamp_tokens);

        return timestamp_log_prob > max_text_token_log_prob;
      }

    };

  }
}
