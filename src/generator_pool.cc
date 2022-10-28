#include "ctranslate2/generator_pool.h"

#include <spdlog/spdlog.h>

namespace ctranslate2 {

  GeneratorPool::GeneratorPool(size_t num_generators_per_device,
                               size_t num_threads_per_generator,
                               const std::string& model_dir,
                               const Device device,
                               const std::vector<int>& device_indices,
                               const ComputeType compute_type,
                               const long max_queued_batches)
    : ReplicaPool(num_generators_per_device,
                  num_threads_per_generator,
                  model_dir,
                  device,
                  device_indices,
                  compute_type,
                  max_queued_batches)
  {
  }

  std::vector<std::future<GenerationResult>>
  GeneratorPool::generate_batch_async(const std::vector<std::vector<std::string>>& start_tokens,
                                      const GenerationOptions& options,
                                      const size_t max_batch_size,
                                      const BatchType batch_type) {
    return post_examples<GenerationResult>(
      load_examples({start_tokens}),
      max_batch_size,
      batch_type,
      [options](models::SequenceGeneratorReplica& generator, const Batch& batch) {
        spdlog::debug("Running batch generation on {} examples", batch.num_examples());
        auto results = generator.generate(batch.get_stream(0), options);
        spdlog::debug("Finished batch generation");
        return results;
      });
  }

  std::vector<std::future<ScoringResult>>
  GeneratorPool::score_batch_async(const std::vector<std::vector<std::string>>& tokens,
                                   const ScoringOptions& options,
                                   const size_t max_batch_size,
                                   const BatchType batch_type) {
    return post_examples<ScoringResult>(
      load_examples({tokens}),
      max_batch_size,
      batch_type,
      [options](models::SequenceGeneratorReplica& generator, const Batch& batch) {
        spdlog::debug("Running batch scoring on {} examples", batch.num_examples());
        auto results = generator.score(batch.get_stream(0), options);
        spdlog::debug("Finished batch scoring");
        return results;
      });
  }

  std::future<StorageView>
  GeneratorPool::forward_batch_async(std::vector<std::vector<std::string>> tokens,
                                     const bool return_log_probs) {
    return post<StorageView>(
      [tokens = std::move(tokens), return_log_probs]
      (models::SequenceGeneratorReplica& generator) {
        return generator.forward(tokens, return_log_probs);
      });
  }

  std::future<StorageView>
  GeneratorPool::forward_batch_async(std::vector<std::vector<size_t>> ids,
                                     const bool return_log_probs) {
    return post<StorageView>(
      [ids = std::move(ids), return_log_probs]
      (models::SequenceGeneratorReplica& generator) {
        return generator.forward(ids, return_log_probs);
      });
  }

  std::future<StorageView>
  GeneratorPool::forward_batch_async(StorageView ids,
                                     StorageView lengths,
                                     const bool return_log_probs) {
    return post<StorageView>(
      [ids = std::move(ids), lengths = std::move(lengths), return_log_probs]
      (models::SequenceGeneratorReplica& generator) {
        return generator.forward(ids, lengths, return_log_probs);
      });
  }

}
