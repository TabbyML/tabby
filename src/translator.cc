#include "opennmt/translator.h"

#include "opennmt/storage_view.h"

namespace opennmt {

  Translator::Translator(const Model& model,
                         size_t max_decoding_steps,
                         size_t beam_size,
                         float length_penalty)
    : _model(model)
    , _encoder(model.make_encoder())
    , _decoder(model.make_decoder())
    , _max_decoding_steps(max_decoding_steps)
    , _beam_size(beam_size)
    , _length_penalty(length_penalty) {
  }

  std::vector<std::string>
  Translator::translate(const std::vector<std::string>& tokens) {
    std::vector<std::vector<std::string>> batch_tokens(1, tokens);
    return translate_batch(batch_tokens)[0];
  }

  std::vector<std::vector<std::string>>
  Translator::translate_batch(const std::vector<std::vector<std::string>>& batch_tokens) {
    size_t batch_size = batch_tokens.size();
    size_t max_length = 0;
    StorageView lengths({batch_size}, DataType::DT_INT32);
    for (size_t i = 0; i < batch_size; ++i) {
      const size_t length = batch_tokens[i].size();
      lengths.at<int32_t>(i) = length;
      max_length = std::max(max_length, length);
    }

    StorageView ids({batch_size, max_length}, DataType::DT_INT32);
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t t = 0; t < batch_tokens[i].size(); ++t) {
        const std::string& token = batch_tokens[i][t];
        ids.at<int32_t>({i, t}) = _model.get_source_vocabulary().to_id(token);
      }
    }

    const auto& encoded = _encoder->encode(ids, lengths);
    _decoder->get_state().reset(encoded, lengths);

    size_t start_token = _model.get_target_vocabulary().to_id("<s>");
    size_t end_token = _model.get_target_vocabulary().to_id("</s>");
    size_t vocabulary_size = _model.get_target_vocabulary().size();
    StorageView sample_from({batch_size, 1}, static_cast<int32_t>(start_token));
    std::vector<std::vector<size_t>> sampled_ids;

    if (_beam_size == 1)
      greedy_decoding(*_decoder,
                      sample_from,
                      end_token,
                      vocabulary_size,
                      _max_decoding_steps,
                      sampled_ids);
    else
      beam_search(*_decoder,
                  sample_from,
                  end_token,
                  vocabulary_size,
                  _max_decoding_steps,
                  _beam_size,
                  _length_penalty,
                  sampled_ids);

    std::vector<std::vector<std::string>> result(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      result[i].reserve(sampled_ids[i].size());
      for (auto id : sampled_ids[i]) {
        result[i].push_back(_model.get_target_vocabulary().to_token(id));
      }
    }
    return result;
  }

}
