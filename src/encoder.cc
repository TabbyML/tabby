#include "ctranslate2/encoder.h"

namespace ctranslate2 {

  std::future<EncoderForwardOutput>
  Encoder::forward_batch_async(std::vector<std::vector<std::string>> tokens,
                               std::vector<std::vector<size_t>> token_type_ids) {
    return post<EncoderForwardOutput>(
      [tokens = std::move(tokens), token_type_ids = std::move(token_type_ids)]
      (models::SequenceEncoderReplica& encoder) {
        return encoder.forward(tokens, token_type_ids);
      });
  }

  std::future<EncoderForwardOutput>
  Encoder::forward_batch_async(std::vector<std::vector<size_t>> ids,
                               std::vector<std::vector<size_t>> token_type_ids) {
    return post<EncoderForwardOutput>(
      [ids = std::move(ids), token_type_ids = std::move(token_type_ids)]
      (models::SequenceEncoderReplica& encoder) {
        return encoder.forward(ids, token_type_ids);
      });
  }

  std::future<EncoderForwardOutput>
  Encoder::forward_batch_async(const StorageView& ids,
                               const StorageView& lengths,
                               std::vector<std::vector<size_t>> token_type_ids) {
    return post<EncoderForwardOutput>(
      [ids = ids.sync_copy(),
       lengths = lengths.sync_copy(),
       token_type_ids = std::move(token_type_ids)]
      (models::SequenceEncoderReplica& encoder) {
        return encoder.forward(ids, lengths, token_type_ids);
      });
  }

}
