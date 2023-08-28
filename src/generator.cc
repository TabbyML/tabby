#include "ctranslate2/generator.h"

#include <spdlog/spdlog.h>

namespace ctranslate2 {

  std::vector<std::future<GenerationResult>>
  Generator::generate_batch_async(const std::vector<std::vector<std::string>>& start_tokens,
                                  const GenerationOptions& options,
                                  const size_t max_batch_size,
                                  const BatchType batch_type) {
    return post_examples<GenerationResult>(
      load_examples({start_tokens}),
      max_batch_size,
      batch_type,
      [options](models::SequenceGeneratorReplica& generator, const Batch& batch) {
        spdlog::debug("Running batch generation on {} examples", batch.num_examples());
        auto results = generator.generate(
          batch.get_stream(0),
          restore_batch_ids_in_callback(options, batch.example_index));
        spdlog::debug("Finished batch generation");
        return results;
      });
  }

  std::vector<std::future<ScoringResult>>
  Generator::score_batch_async(const std::vector<std::vector<std::string>>& tokens,
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
  Generator::forward_batch_async(std::vector<std::vector<std::string>> tokens,
                                 const bool return_log_probs) {
    return post<StorageView>(
      [tokens = std::move(tokens), return_log_probs]
      (models::SequenceGeneratorReplica& generator) {
        return generator.forward(tokens, return_log_probs);
      });
  }

  std::future<StorageView>
  Generator::forward_batch_async(std::vector<std::vector<size_t>> ids,
                                 const bool return_log_probs) {
    return post<StorageView>(
      [ids = std::move(ids), return_log_probs]
      (models::SequenceGeneratorReplica& generator) {
        return generator.forward(ids, return_log_probs);
      });
  }

  std::future<StorageView>
  Generator::forward_batch_async(StorageView ids,
                                 StorageView lengths,
                                 const bool return_log_probs) {
    return post<StorageView>(
      [ids = std::move(ids), lengths = std::move(lengths), return_log_probs]
      (models::SequenceGeneratorReplica& generator) {
        return generator.forward(ids, lengths, return_log_probs);
      });
  }

}
