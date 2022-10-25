#pragma once

#include "replica_pool.h"
#include "models/language_model.h"

namespace ctranslate2 {

  // GeneratorPool is the high-level class for running generation with language models.
  // It supports parallel and asynchronous generation.
  class GeneratorPool : public ReplicaPool<models::SequenceGeneratorReplica> {
  public:
    GeneratorPool(size_t num_generators_per_device,
                  size_t num_threads_per_generator,
                  const std::string& model_dir,
                  const Device device,
                  const std::vector<int>& device_indices = {0},
                  const ComputeType compute_type = ComputeType::DEFAULT,
                  const long max_queued_batches = 0);

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
  };

}
