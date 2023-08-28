#pragma once

#include "replica_pool.h"
#include "models/language_model.h"

namespace ctranslate2 {

  // Encoder is the high-level class to embed texts with language models.
  class Encoder : public ReplicaPool<models::SequenceEncoderReplica> {
  public:
    using ReplicaPool::ReplicaPool;

    std::future<EncoderForwardOutput>
    forward_batch_async(std::vector<std::vector<std::string>> tokens,
                        std::vector<std::vector<size_t>> token_type_ids = {});

    std::future<EncoderForwardOutput>
    forward_batch_async(std::vector<std::vector<size_t>> ids,
                        std::vector<std::vector<size_t>> token_type_ids = {});

    std::future<EncoderForwardOutput>
    forward_batch_async(const StorageView& ids,
                        const StorageView& lengths,
                        std::vector<std::vector<size_t>> token_type_ids = {});
  };

}
