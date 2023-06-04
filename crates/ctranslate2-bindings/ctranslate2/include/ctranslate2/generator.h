#pragma once

#include "replica_pool.h"
#include "models/language_model.h"

namespace ctranslate2 {

  // Generator is the high-level class for running generation with language models.
  // It supports parallel and asynchronous generation.
  class Generator : public ReplicaPool<models::SequenceGeneratorReplica> {
  public:
    using ReplicaPool::ReplicaPool;

    std::vector<std::future<GenerationResult>>
    generate_batch_async(const std::vector<std::vector<std::string>>& start_tokens,
                         const GenerationOptions& options = GenerationOptions(),
                         const size_t max_batch_size = 0,
                         const BatchType batch_type = BatchType::Examples);

    std::vector<std::future<ScoringResult>>
    score_batch_async(const std::vector<std::vector<std::string>>& tokens,
                      const ScoringOptions& options = ScoringOptions(),
                      const size_t max_batch_size = 0,
                      const BatchType batch_type = BatchType::Examples);

    std::future<StorageView>
    forward_batch_async(std::vector<std::vector<std::string>> tokens,
                        const bool return_log_probs);

    std::future<StorageView>
    forward_batch_async(std::vector<std::vector<size_t>> ids,
                        const bool return_log_probs);

    std::future<StorageView>
    forward_batch_async(StorageView ids,
                        StorageView lengths,
                        const bool return_log_probs);
  };

}
