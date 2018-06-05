#include "opennmt/translator.h"

#include <iostream>

#include "opennmt/storage_view.h"

namespace opennmt {

  void translate(const std::vector<std::vector<std::string> >& input_tokens,
                 const Vocabulary& vocabulary,
                 Encoder& encoder,
                 Decoder& decoder,
                 size_t beam_size) {
    size_t batch_size = input_tokens.size();
    size_t max_length = 0;
    StorageView lengths({batch_size}, DataType::DT_INT32);
    for (size_t i = 0; i < batch_size; ++i) {
      const size_t length = input_tokens[i].size();
      lengths.at<int32_t>(i) = length;
      max_length = std::max(max_length, length);
    }

    StorageView ids({batch_size, max_length}, DataType::DT_INT32);
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t t = 0; t < input_tokens[i].size(); ++t) {
        ids.at<int32_t>({i, t}) = vocabulary.to_id(input_tokens[i][t]);
      }
    }

    const auto& encoded = encoder.encode(ids, lengths);

    StorageView sample_from({batch_size, 1}, static_cast<int32_t>(vocabulary.to_id("<s>")));
    std::vector<std::vector<size_t>> sampled_ids;

    decoder.get_state().reset(encoded, lengths);
    if (beam_size == 1)
      greedy_decoding(decoder, sample_from, 2, vocabulary.size(), 200, sampled_ids);
    else
      beam_search(decoder, sample_from, 2, 5, 0.6, vocabulary.size(), 200, sampled_ids);

    for (size_t i = 0; i < batch_size; ++i) {
      for (auto id : sampled_ids[i]) {
        std::cout << " " << vocabulary.to_token(id);
      }
      std::cout << std::endl;
    }
  }

}
