#include "ctranslate2/translator.h"

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {

  Translator::Translator(const std::shared_ptr<Model>& model,
                         size_t max_decoding_steps,
                         size_t beam_size,
                         float length_penalty,
                         const std::string& vocabulary_map)
    : _model(model)
    , _encoder(_model->make_encoder())
    , _decoder(_model->make_decoder())
    , _max_decoding_steps(max_decoding_steps)
    , _beam_size(beam_size)
    , _length_penalty(length_penalty) {
    if (!vocabulary_map.empty())
      _vocabulary_map.reset(new VocabularyMap(vocabulary_map, _model->get_target_vocabulary()));
  }

  Translator::Translator(const Translator& other)
    : _model(other._model)
    , _vocabulary_map(other._vocabulary_map)
    , _encoder(_model->make_encoder())  // Makes a new graph to ensure thread safety.
    , _decoder(_model->make_decoder())  // Same here.
    , _max_decoding_steps(other._max_decoding_steps)
    , _beam_size(other._beam_size)
    , _length_penalty(other._length_penalty) {
  }

  std::vector<std::string>
  Translator::translate(const std::vector<std::string>& tokens) {
    std::vector<std::vector<std::string>> batch_tokens(1, tokens);
    return translate_batch(batch_tokens)[0];
  }

  std::vector<std::vector<std::string>>
  Translator::translate_batch(const std::vector<std::vector<std::string>>& batch_tokens) {
    size_t batch_size = batch_tokens.size();

    // Record lengths and maximum length.
    size_t max_length = 0;
    StorageView lengths({batch_size}, DataType::DT_INT32);
    for (size_t i = 0; i < batch_size; ++i) {
      const size_t length = batch_tokens[i].size();
      lengths.at<int32_t>(i) = length;
      max_length = std::max(max_length, length);
    }

    // Convert tokens to ids.
    StorageView ids({batch_size, max_length}, DataType::DT_INT32);
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t t = 0; t < batch_tokens[i].size(); ++t) {
        const std::string& token = batch_tokens[i][t];
        ids.at<int32_t>({i, t}) = _model->get_source_vocabulary().to_id(token);
      }
    }

    // Encode sequence.
    static thread_local StorageView encoded;
    _encoder->encode(ids, lengths, encoded);

    // Reset decoder states based on the encoder outputs.
    _decoder->get_state().reset(encoded, lengths);

    // If set, extract the subset of candidates to generate.
    StorageView candidates(DataType::DT_INT32);
    if (_vocabulary_map) {
      auto candidates_vec = _vocabulary_map->get_candidates<int32_t>(batch_tokens);
      candidates.resize({candidates_vec.size()});
      candidates.copy_from(candidates_vec.data(), candidates_vec.size());
    }

    // Decode.
    size_t start_token = _model->get_target_vocabulary().to_id("<s>");
    size_t end_token = _model->get_target_vocabulary().to_id("</s>");
    StorageView sample_from({batch_size, 1}, static_cast<int32_t>(start_token));
    std::vector<std::vector<size_t>> sampled_ids;
    if (_beam_size == 1)
      greedy_decoding(*_decoder,
                      sample_from,
                      candidates,
                      end_token,
                      _max_decoding_steps,
                      sampled_ids);
    else
      beam_search(*_decoder,
                  sample_from,
                  candidates,
                  end_token,
                  _max_decoding_steps,
                  _beam_size,
                  _length_penalty,
                  sampled_ids);

    // Build result.
    std::vector<std::vector<std::string>> result(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      result[i].reserve(sampled_ids[i].size());
      for (auto id : sampled_ids[i]) {
        result[i].push_back(_model->get_target_vocabulary().to_token(id));
      }
    }
    return result;
  }

}
