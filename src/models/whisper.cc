#include "ctranslate2/models/whisper.h"

#include <algorithm>

#include "ctranslate2/decoding.h"
#include "ctranslate2/models/model_factory.h"

namespace ctranslate2 {
  namespace models {

    static auto register_whisper = register_model<WhisperModel>("WhisperSpec");

    const Vocabulary& WhisperModel::get_vocabulary() const {
      return *_vocabulary;
    }

    void WhisperModel::initialize(ModelReader& model_reader) {
      VocabularyInfo vocab_info;
      vocab_info.unk_token = config["unk_token"];
      vocab_info.bos_token = config["bos_token"];
      vocab_info.eos_token = config["eos_token"];
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
      , _decoder(std::make_unique<layers::TransformerDecoder>(*model, "decoder"))
    {
    }

    StorageView WhisperReplica::encode(const StorageView& features) {
      const Device device = _model->device();

      StorageView encoder_output(_encoder->output_type(), device);
      if (features.device() == device)
        (*_encoder)(features, encoder_output);
      else
        (*_encoder)(features.to(device), encoder_output);

      return encoder_output;
    }

    std::vector<GenerationResult>
    WhisperReplica::generate(const StorageView& features,
                             const std::vector<std::vector<std::string>>& prompts,
                             const WhisperOptions& options) {
      const auto& vocabulary = _model->get_vocabulary();
      return generate(features, vocabulary.to_ids(prompts), options);
    }

    std::vector<GenerationResult>
    WhisperReplica::generate(const StorageView& features,
                             std::vector<std::vector<size_t>> prompts,
                             const WhisperOptions& options) {
      PROFILE("WhisperReplica::generate");
      if (prompts.empty())
        return {};

      const auto& vocabulary = _model->get_vocabulary();
      const auto scoped_device_setter = _model->get_scoped_device_setter();

      DecodingOptions decoding_options;
      decoding_options.beam_size = options.beam_size;
      decoding_options.length_penalty = options.length_penalty;
      decoding_options.max_length = options.max_length;
      decoding_options.sampling_topk = options.sampling_topk;
      decoding_options.sampling_temperature = options.sampling_temperature;
      decoding_options.num_hypotheses = options.num_hypotheses;
      decoding_options.return_scores = options.return_scores;
      for (const auto& id : _model->config["suppress_ids"])
        decoding_options.disable_ids.push_back(id);
      for (const auto& id : _model->config["suppress_ids_begin"])
        decoding_options.disable_ids_begin.push_back(id);

      layers::DecoderState state = _decoder->initial_state();
      state.emplace("memory", encode(features));

      _decoder->update_output_layer(_model->preferred_size_multiple());

      const size_t sot = _model->config["decoder_start_id"];
      for (auto& prompt : prompts)
        prompt.insert(prompt.begin(), sot);

      std::vector<DecodingResult> results = decode(*_decoder,
                                                   state,
                                                   prompts,
                                                   vocabulary.eos_id(),
                                                   decoding_options);

      std::vector<GenerationResult> final_results;
      final_results.reserve(results.size());
      for (auto& result : results) {
        GenerationResult final_result;
        final_result.sequences = vocabulary.to_tokens(result.hypotheses);
        final_result.sequences_ids = std::move(result.hypotheses);
        final_result.scores = std::move(result.scores);
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

      const int32_t sot = _model->config["decoder_start_id"];
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


    Whisper::Whisper(size_t num_replicas_per_device,
                     size_t num_threads_per_replica,
                     const std::string& path,
                     const Device device,
                     const std::vector<int>& device_indices,
                     const ComputeType compute_type,
                     const long max_queued_batches)
      : ReplicaPool(num_replicas_per_device,
                    num_threads_per_replica,
                    path,
                    device,
                    device_indices,
                    compute_type,
                    max_queued_batches)
    {
    }

    std::vector<std::future<GenerationResult>>
    Whisper::generate(StorageView features,
                      std::vector<std::vector<std::string>> prompts,
                      WhisperOptions options) {
      const size_t batch_size = features.dim(0);
      return post_batch<GenerationResult>(
        [features = std::move(features), prompts = std::move(prompts), options]
        (WhisperReplica& replica) {
          return replica.generate(features, prompts, options);
        },
        batch_size);
    }

    std::vector<std::future<GenerationResult>>
    Whisper::generate(StorageView features,
                      std::vector<std::vector<size_t>> prompts,
                      WhisperOptions options) {
      const size_t batch_size = features.dim(0);
      return post_batch<GenerationResult>(
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

  }
}
