#include "ctranslate2/models/whisper.h"

#include <algorithm>

#include "ctranslate2/decoding.h"
#include "ctranslate2/models/model_factory.h"

#include "dispatch.h"
#include "dtw.h"

#ifdef CT2_WITH_CUDA
#  include "cuda/utils.h"
#endif

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
      const auto& vocabulary = model->get_vocabulary();
      _sot_id = vocabulary.bos_id();
      _eot_id = vocabulary.eos_id();
      _no_timestamps_id = vocabulary.to_id("<|notimestamps|>");
      _no_speech_id = vocabulary.to_id("<|nospeech|>");
      if (_no_speech_id == vocabulary.unk_id())
        _no_speech_id = vocabulary.to_id("<|nocaptions|>");
      _is_multilingual = vocabulary.size() == 51865;
    }

    StorageView WhisperReplica::encode(StorageView features, const bool to_cpu) {
      PROFILE("WhisperReplica::encode");

#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const Device device = _model->device();
      const DataType dtype = _encoder->output_type();
      features.move_to(device, dtype);

      StorageView encoder_output(dtype, device);
      (*_encoder)(features, encoder_output);

      if (to_cpu) {
        if (device != Device::CPU)
          encoder_output = encoder_output.to(Device::CPU);
        return encoder_output;
      }

      // Ensure all operations are finished before returning the output.
      synchronize_stream(device);

      return encoder_output;
    }

    StorageView WhisperReplica::maybe_encode(StorageView features) {
      const Device device = _model->device();
      const DataType dtype = _encoder->output_type();

      features.move_to(device, dtype);

      // Already encoded.
      if (features.dim(-1) == _encoder->output_size()
          && features.dim(-2) == _encoder->output_time())
        return features;

      StorageView encoder_output(dtype, device);
      (*_encoder)(features, encoder_output);
      return encoder_output;
    }

    std::vector<WhisperGenerationResult>
    WhisperReplica::generate(StorageView features,
                             const std::vector<std::vector<std::string>>& prompts,
                             const WhisperOptions& options) {
      const auto& vocabulary = _model->get_vocabulary();
      return generate(std::move(features), vocabulary.to_ids(prompts), options);
    }

    static std::vector<float> get_no_speech_probs_from_logits(const StorageView& logits,
                                                              const size_t no_speech_id) {
      const Device device = logits.device();
      const DataType dtype = logits.dtype();

      StorageView probs(dtype, device);
      ops::SoftMax()(logits, probs);

      StorageView gather_ids({probs.dim(0)}, int32_t(no_speech_id), device);
      StorageView no_speech_probs(dtype, device);
      ops::Gather(/*axis=*/1, /*batch_dims=*/1)(probs, gather_ids, no_speech_probs);

      if (no_speech_probs.dtype() != DataType::FLOAT32)
        no_speech_probs = no_speech_probs.to_float32();
      return no_speech_probs.to_vector<float>();
    }

    static size_t get_sot_index(const std::vector<size_t>& prompt, const size_t sot_id) {
      const auto sot_it = std::find(prompt.begin(), prompt.end(), sot_id);
      if (sot_it == prompt.end())
          throw std::invalid_argument("<|startoftranscript|> token was not found in the prompt");

      return std::distance(prompt.begin(), sot_it);
    }

    static size_t get_prompt_length(const std::vector<size_t>& prompt,
                                    const size_t sot_id,
                                    const size_t no_timestamps_id) {
      size_t index = get_sot_index(prompt, sot_id);
      while (index < prompt.size() && prompt[index] >= sot_id && prompt[index] <= no_timestamps_id)
        index++;
      return index;
    }

    static void check_prompts(const std::vector<std::vector<size_t>>& prompts,
                              const size_t sot_id,
                              const size_t no_timestamps_id,
                              size_t& sot_index,
                              size_t& prompt_length) {
      bool first = true;

      for (const auto& prompt : prompts) {
        const auto batch_sot_index = get_sot_index(prompt, sot_id);
        const auto batch_prompt_length = get_prompt_length(prompt, sot_id, no_timestamps_id);

        if (first) {
          sot_index = batch_sot_index;
          prompt_length = batch_prompt_length;
        } else if (batch_sot_index != sot_index) {
          throw std::invalid_argument("The generate method currently requires the "
                                      "<|startoftranscript|> token to be at the same position "
                                      "in all batches. To work around this limitation, "
                                      "simply adapt the number of previous text tokens in each "
                                      "batch.");
        } else if (batch_prompt_length != prompt_length) {
          throw std::invalid_argument("The generate method currently requires each batch to have "
                                      "the same number of task tokens after <|startoftranscript|>.");
        }

        first = false;
      }
    }

    class ApplyTimestampRules;

    class GetNoSpeechProbs : public LogitsProcessor {
    private:
      const size_t _no_speech_id;
      std::vector<float> _no_speech_probs;

    public:
      GetNoSpeechProbs(const size_t no_speech_id)
        : _no_speech_id(no_speech_id)
      {
      }

      const std::vector<float>& get_no_speech_probs() const {
        return _no_speech_probs;
      }

      bool apply_first() const override {
        return true;
      }

      void apply(dim_t step,
                 StorageView& logits,
                 DisableTokens&,
                 const StorageView&,
                 const std::vector<dim_t>& batch_offset,
                 const std::vector<std::vector<size_t>>*) override {
        if (step == 0) {
          const auto no_speech_probs = get_no_speech_probs_from_logits(logits, _no_speech_id);

          const size_t batch_size = batch_offset.size();
          const size_t beam_size = logits.dim(0) / batch_size;

          _no_speech_probs.reserve(batch_size);
          for (size_t i = 0; i < batch_size; ++i)
            _no_speech_probs.emplace_back(no_speech_probs[i * beam_size]);
        }
      }
    };

    std::vector<WhisperGenerationResult>
    WhisperReplica::generate(StorageView features,
                             const std::vector<std::vector<size_t>>& prompts,
                             const WhisperOptions& options) {
      PROFILE("WhisperReplica::generate");
      if (prompts.empty())
        return {};

#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

      size_t sot_index = 0;
      size_t prompt_length = 0;  // Length of the prompt before the text tokens.
      check_prompts(prompts, _sot_id, _no_timestamps_id, sot_index, prompt_length);

      const auto& vocabulary = _model->get_vocabulary();
      const auto scoped_device_setter = _model->get_scoped_device_setter();

      layers::DecoderState state = _decoder->initial_state();
      state.emplace("memory", maybe_encode(std::move(features)));

      _decoder->update_output_layer(_model->preferred_size_multiple());

      const bool sot_is_start_token = (sot_index == prompt_length - 1);
      std::vector<std::vector<size_t>> start_tokens;
      std::vector<float> no_speech_probs;
      dim_t start_step = 0;

      if (prompt_length == 1) {
        start_tokens = prompts;

      } else {
        std::vector<std::vector<size_t>> prompt_tokens;
        prompt_tokens.reserve(prompts.size());
        start_tokens.reserve(prompts.size());
        for (const auto& prompt : prompts) {
          prompt_tokens.emplace_back(prompt.begin(), prompt.begin() + prompt_length - 1);
          start_tokens.emplace_back(prompt.begin() + prompt_length - 1, prompt.end());
        }

        const Device device = _decoder->device();
        const DataType dtype = _decoder->output_type();
        const StorageView inputs = layers::make_sequence_inputs(prompt_tokens, device);

        // Initialize the decoder state with the prompt.
        if (!options.return_no_speech_prob || sot_is_start_token)
          _decoder->forward_prompt(inputs, state);
        else {
          StorageView outputs(dtype, device);
          _decoder->forward_prompt(inputs, state, &outputs);

          // Get the probability of the no speech token at the start of transcript step.
          StorageView sot_index_batch({inputs.dim(0)}, int32_t(sot_index), device);
          StorageView logits(dtype, device);
          _decoder->compute_logits_for_steps(outputs, sot_index_batch, logits);
          no_speech_probs = get_no_speech_probs_from_logits(logits, _no_speech_id);
        }

        start_step = inputs.dim(1);
      }

      const dim_t total_max_length = options.max_length;

      DecodingOptions decoding_options;
      decoding_options.start_step = start_step;
      decoding_options.beam_size = options.beam_size;
      decoding_options.patience = options.patience;
      decoding_options.length_penalty = options.length_penalty;
      decoding_options.repetition_penalty = options.repetition_penalty;
      decoding_options.no_repeat_ngram_size = options.no_repeat_ngram_size;
      decoding_options.max_length = std::min(total_max_length / 2, total_max_length - start_step);
      decoding_options.sampling_topk = options.sampling_topk;
      decoding_options.sampling_temperature = options.sampling_temperature;
      decoding_options.num_hypotheses = options.num_hypotheses;
      decoding_options.return_scores = options.return_scores;
      decoding_options.include_eos_in_hypotheses = false;

      for (const auto& id : options.suppress_tokens) {
        if (id >= 0)
          decoding_options.disable_ids.push_back(id);
        else if (id == -1) {
          for (const auto& default_id : _model->config["suppress_ids"])
            decoding_options.disable_ids.push_back(default_id);
        }
      }

      if (options.suppress_blank) {
        for (const auto& id : _model->config["suppress_ids_begin"])
          decoding_options.disable_ids_begin.push_back(id);
      }

      std::shared_ptr<GetNoSpeechProbs> no_speech_probs_processor;
      if (options.return_no_speech_prob && sot_is_start_token) {
        // If SOT is the start token, we need to get the no speech prob in the first decoding loop.
        no_speech_probs_processor = std::make_shared<GetNoSpeechProbs>(_no_speech_id);
        decoding_options.logits_processors.emplace_back(no_speech_probs_processor);
      }

      if (prompts[0][prompt_length - 1] != _no_timestamps_id) {
        const size_t timestamp_begin_id = _no_timestamps_id + 1;
        const size_t timestamp_end_id = vocabulary.size() - 1;
        const size_t max_initial_timestamp_id = timestamp_begin_id + options.max_initial_timestamp_index;
        decoding_options.logits_processors.emplace_back(
          std::make_shared<ApplyTimestampRules>(_eot_id,
                                                _no_timestamps_id,
                                                timestamp_begin_id,
                                                timestamp_end_id,
                                                max_initial_timestamp_id));
      }

      std::vector<DecodingResult> results = decode(*_decoder,
                                                   state,
                                                   start_tokens,
                                                   _eot_id,
                                                   decoding_options);

      if (no_speech_probs_processor)
        no_speech_probs = no_speech_probs_processor->get_no_speech_probs();

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

    std::vector<WhisperAlignmentResult>
    WhisperReplica::align(StorageView features,
                          const std::vector<size_t>& start_sequence,
                          const std::vector<std::vector<size_t>>& text_tokens,
                          dim_t num_frames,
                          dim_t median_filter_width) {
      PROFILE("WhisperReplica::align");

      const auto alignment_heads = _model->config.find("alignment_heads");
      if (alignment_heads == _model->config.end())
        throw std::runtime_error("The model configuration does not contain the field "
                                 "'alignment_heads' which lists the cross-attention heads "
                                 "that are highly correlated to the word-level timing. "
                                 "Please reconvert this model with the current version "
                                 "of ctranslate2.");

      _decoder->set_alignment_heads(alignment_heads->get<std::vector<std::pair<dim_t, dim_t>>>());

      const dim_t batch_size = text_tokens.size();

      std::vector<std::vector<size_t>> input_tokens;
      std::vector<std::vector<size_t>> output_tokens;
      input_tokens.reserve(batch_size);
      output_tokens.reserve(batch_size);

      for (const auto& text_sequence : text_tokens) {
        std::vector<size_t> input_sequence = start_sequence;
        input_sequence.push_back(_no_timestamps_id);
        input_sequence.insert(input_sequence.end(), text_sequence.begin(), text_sequence.end());
        input_sequence.push_back(_eot_id);

        std::vector<size_t> output_sequence(input_sequence.begin() + 1, input_sequence.end());
        output_sequence.push_back(0);

        input_tokens.emplace_back(std::move(input_sequence));
        output_tokens.emplace_back(std::move(output_sequence));
      }

      const auto scoped_device_setter = _model->get_scoped_device_setter();

#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

      layers::DecoderState state = _decoder->initial_state(/*iterative_decoding=*/false);
      state.emplace("memory", maybe_encode(std::move(features)));

      _decoder->update_output_layer(_model->preferred_size_multiple());

      const DataType dtype = _decoder->output_type();
      const Device device = _decoder->device();

      StorageView lengths(DataType::INT32, device);
      StorageView input_ids = layers::make_sequence_inputs(input_tokens,
                                                           device,
                                                           1,
                                                           &lengths);
      StorageView output_ids = layers::make_sequence_inputs(output_tokens, device);

      StorageView logits(dtype, device);
      StorageView attention_weights(dtype, device);
      (*_decoder)(input_ids, lengths, state, logits, &attention_weights);

      StorageView token_probs(dtype, device);

      {
        // Get the probabilities of the text tokens.
        StorageView text_vocab_size({logits.dim(0), logits.dim(1)}, int32_t(_eot_id), device);
        ops::SoftMax()(logits, text_vocab_size, logits);
        StorageView probs = std::move(logits);

        ops::Gather(/*axis=*/-1, /*batch_dims=*/2)(probs, output_ids, token_probs);
      }

      StorageView weights;

      {
        // Process the attention probs.
        const dim_t frames_dim = 3;
        const dim_t total_frames = attention_weights.dim(frames_dim);

        num_frames /= 2;  // The second convolution layer uses a stride of 2.

        if (num_frames < total_frames) {
          StorageView content(dtype, device);
          StorageView padding(dtype, device);

          const ops::Split split_op(frames_dim, {num_frames, total_frames - num_frames});
          split_op(attention_weights, content, padding);

          attention_weights = std::move(content);
        }

        ops::SoftMax()(attention_weights);

        // The remaining operations are not implemented on GPU, so move back to CPU.
        attention_weights.move_to(Device::CPU, DataType::FLOAT32);

        ops::LayerNorm(-2, 0)(attention_weights);

        const ops::MedianFilter median_filter_op(median_filter_width);
        StorageView median_filter;
        median_filter_op(attention_weights, median_filter);
        ops::Mean(1)(median_filter, weights);
      }

      token_probs.move_to(Device::CPU, DataType::FLOAT32);

      std::vector<WhisperAlignmentResult> results;
      results.reserve(batch_size);

      for (dim_t b = 0; b < batch_size; ++b) {
        WhisperAlignmentResult result;

        const dim_t length = text_tokens[b].size();
        const dim_t offset = start_sequence.size();

        StorageView matrix(
          Shape{weights.dim(1) - offset - 1, weights.dim(2)},
          weights.index<float>({b, offset, 0}));

        result.alignments = negative_dtw(matrix);

        for (dim_t t = 0; t < length; ++t)
          result.text_token_probs.emplace_back(token_probs.at<float>({b, offset + t}));

        results.emplace_back(std::move(result));
      }

      return results;
    }

    std::vector<std::vector<std::pair<std::string, float>>>
    WhisperReplica::detect_language(StorageView features) {
      if (!is_multilingual())
        throw std::runtime_error("detect_language can only be called on multilingual models");

      PROFILE("WhisperReplica::detect_language");

#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

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
      state.emplace("memory", maybe_encode(std::move(features)));

      StorageView logits(_decoder->output_type(), device);
      StorageView lang_probs(logits.dtype(), device);
      (*_decoder)(0, start_ids, state, &logits);
      ops::Gather(/*axis=*/-1, /*batch_dims=*/1)(logits, score_ids, lang_probs);
      ops::SoftMax()(lang_probs);

      if (lang_probs.dtype() != DataType::FLOAT32)
        lang_probs = lang_probs.to_float32();
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


    bool Whisper::is_multilingual() const {
      const auto& replica = get_first_replica();
      return replica.is_multilingual();
    }

    std::future<StorageView> Whisper::encode(const StorageView& features, const bool to_cpu) {
      return post<StorageView>([features = features.sync_copy(), to_cpu](WhisperReplica& replica) {
        return replica.encode(std::move(features), to_cpu);
      });
    }

    std::vector<std::future<WhisperGenerationResult>>
    Whisper::generate(const StorageView& features,
                      std::vector<std::vector<std::string>> prompts,
                      WhisperOptions options) {
      const size_t batch_size = features.dim(0);
      return post_batch<WhisperGenerationResult>(
        [features = features.sync_copy(), prompts = std::move(prompts), options]
        (WhisperReplica& replica) {
          return replica.generate(std::move(features), prompts, options);
        },
        batch_size);
    }

    std::vector<std::future<WhisperGenerationResult>>
    Whisper::generate(const StorageView& features,
                      std::vector<std::vector<size_t>> prompts,
                      WhisperOptions options) {
      const size_t batch_size = features.dim(0);
      return post_batch<WhisperGenerationResult>(
        [features = features.sync_copy(), prompts = std::move(prompts), options]
        (WhisperReplica& replica) {
          return replica.generate(std::move(features), prompts, options);
        },
        batch_size);
    }

    std::vector<std::future<std::vector<std::pair<std::string, float>>>>
    Whisper::detect_language(const StorageView& features) {
      const size_t batch_size = features.dim(0);
      return post_batch<std::vector<std::pair<std::string, float>>>(
        [features = features.sync_copy()](WhisperReplica& replica) {
          return replica.detect_language(std::move(features));
        },
        batch_size);
    }

    std::vector<std::future<WhisperAlignmentResult>>
    Whisper::align(const StorageView& features,
                   std::vector<size_t> start_sequence,
                   std::vector<std::vector<size_t>> text_tokens,
                   dim_t num_frames,
                   dim_t median_filter_width) {
      const size_t batch_size = features.dim(0);
      return post_batch<WhisperAlignmentResult>(
        [features = features.sync_copy(),
         start_sequence = std::move(start_sequence),
         text_tokens = std::move(text_tokens),
         num_frames,
         median_filter_width]
        (WhisperReplica& replica) {
          return replica.align(std::move(features),
                               start_sequence,
                               text_tokens,
                               num_frames,
                               median_filter_width);
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

    public:
      ApplyTimestampRules(const size_t eot_id,
                          const size_t no_timestamps_id,
                          const size_t timestamp_begin_id,
                          const size_t timestamp_end_id,
                          const size_t max_initial_timestamp_id)
        : _eot_id(eot_id)
        , _no_timestamps_id(no_timestamps_id)
        , _timestamp_begin_id(timestamp_begin_id)
        , _timestamp_end_id(timestamp_end_id)
        , _max_initial_timestamp_id(max_initial_timestamp_id)
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
                check_timestamps_prob_for_batch.push_back(batch_id);
              }
            } else {
              check_timestamps_prob_for_batch.push_back(batch_id);
            }

            // Timestamps shouldn't decrease: forbid timestamp tokens smaller than the last.
            for (dim_t t = step - 1; t >= sample_begin; --t) {
              const size_t token = sequences.at<int32_t>({batch_id, t});

              if (token >= _timestamp_begin_id) {
                for (size_t i = _timestamp_begin_id; i < token; ++i)
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
            else if (log_probs.dtype() == DataType::FLOAT32)
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
