#include "ctranslate2/translator.h"

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {

  Translator::Translator(const std::string& model_dir, Device device)
    : _model(models::ModelFactory::load(model_dir, device))
    , _encoder(_model->make_encoder())
    , _decoder(_model->make_decoder()) {
  }

  Translator::Translator(const std::shared_ptr<models::Model>& model)
    : _model(model)
    , _encoder(_model->make_encoder())
    , _decoder(_model->make_decoder()) {
  }

  Translator::Translator(const Translator& other)
    : _model(other._model)
    , _encoder(_model->make_encoder())  // Makes a new graph to ensure thread safety.
    , _decoder(_model->make_decoder()) { // Same here.
  }

  TranslationResult
  Translator::translate(const std::vector<std::string>& tokens) {
    TranslationOptions options;
    return translate(tokens, options);
  }

  TranslationResult
  Translator::translate(const std::vector<std::string>& tokens,
                        const TranslationOptions& options) {
    std::vector<std::vector<std::string>> batch_tokens(1, tokens);
    return translate_batch(batch_tokens, options)[0];
  }

  std::vector<TranslationResult>
  Translator::translate_batch(const std::vector<std::vector<std::string>>& batch_tokens) {
    TranslationOptions options;
    return translate_batch(batch_tokens, options);
  }

  std::vector<TranslationResult>
  Translator::translate_batch(const std::vector<std::vector<std::string>>& batch_tokens,
                              const TranslationOptions& options) {
    const auto& source_vocab = _model->get_source_vocabulary();
    const auto& target_vocab = _model->get_target_vocabulary();
    const auto& vocab_map = _model->get_vocabulary_map();
    auto& encoder = *_encoder;
    auto& decoder = *_decoder;

    // Check options.
    if (options.num_hypotheses > options.beam_size)
      throw std::invalid_argument("The number of hypotheses can not be greater than the beam size");
    if (options.use_vmap && vocab_map.empty())
      throw std::invalid_argument("use_vmap is set but the model does not include a vocabulary map");
    if (options.min_decoding_length > options.max_decoding_length)
      throw std::invalid_argument("min_decoding_length is greater than max_decoding_length");

    size_t batch_size = batch_tokens.size();

    // Record lengths and maximum length.
    size_t max_length = 0;
    StorageView lengths_host({batch_size}, DataType::DT_INT32);
    for (size_t i = 0; i < batch_size; ++i) {
      const size_t length = batch_tokens[i].size();
      lengths_host.at<int32_t>(i) = length;
      max_length = std::max(max_length, length);
    }

    // Convert tokens to ids.
    StorageView ids_host({batch_size, max_length}, DataType::DT_INT32);
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t t = 0; t < batch_tokens[i].size(); ++t) {
        const std::string& token = batch_tokens[i][t];
        ids_host.at<int32_t>({i, t}) = source_vocab.to_id(token);
      }
    }

    auto device = _model->device();
    StorageView ids = ids_host.to(device);
    StorageView lengths = lengths_host.to(device);

    // Encode sequence.
    StorageView encoded(device);
    encoder(ids, lengths, encoded);

    // If set, extract the subset of candidates to generate.
    StorageView candidates(DataType::DT_INT32, device);
    if (options.use_vmap && !vocab_map.empty()) {
      auto candidates_vec = vocab_map.get_candidates<int32_t>(batch_tokens);
      candidates.resize({candidates_vec.size()});
      candidates.copy_from(candidates_vec.data(), candidates_vec.size(), Device::CPU);
    }

    // Decode.
    size_t start_token = target_vocab.to_id(Vocabulary::bos_token);
    size_t end_token = target_vocab.to_id(Vocabulary::eos_token);
    StorageView sample_from({batch_size}, static_cast<int32_t>(start_token));
    std::vector<std::vector<std::vector<size_t>>> sampled_ids;
    std::vector<std::vector<float>> scores;
    if (options.beam_size == 1)
      greedy_decoding(decoder,
                      sample_from,
                      candidates,
                      encoded,
                      lengths,
                      end_token,
                      options.max_decoding_length,
                      options.min_decoding_length,
                      sampled_ids,
                      scores);
    else
      beam_search(decoder,
                  sample_from,
                  candidates,
                  encoded,
                  lengths,
                  end_token,
                  options.max_decoding_length,
                  options.min_decoding_length,
                  options.beam_size,
                  options.num_hypotheses,
                  options.length_penalty,
                  sampled_ids,
                  scores);

    // Build results.
    std::vector<TranslationResult> results;
    results.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i)
      results.emplace_back(sampled_ids[i], scores[i], target_vocab);
    return results;
  }

  Device Translator::device() const {
    return _model->device();
  }

}
