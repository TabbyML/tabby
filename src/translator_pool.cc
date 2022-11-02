#include "ctranslate2/translator_pool.h"

#include <spdlog/spdlog.h>

namespace ctranslate2 {

  TranslatorPool::TranslatorPool(size_t num_translators,
                                 size_t num_threads_per_translator,
                                 const std::string& model_dir,
                                 const Device device,
                                 const int device_index,
                                 const ComputeType compute_type,
                                 const long max_queued_batches)
    : ReplicaPool(num_translators,
                  num_threads_per_translator,
                  model_dir,
                  device,
                  {device_index},
                  compute_type,
                  max_queued_batches)
  {
  }

  TranslatorPool::TranslatorPool(size_t num_translators,
                                 size_t num_threads_per_translator,
                                 models::ModelReader& model_reader,
                                 const Device device,
                                 const int device_index,
                                 const ComputeType compute_type,
                                 const long max_queued_batches)
    : ReplicaPool(num_translators,
                  num_threads_per_translator,
                  model_reader,
                  device,
                  {device_index},
                  compute_type,
                  max_queued_batches)
  {
  }

  TranslatorPool::TranslatorPool(size_t num_translators_per_device,
                                 size_t num_threads_per_translator,
                                 const std::string& model_dir,
                                 const Device device,
                                 const std::vector<int>& device_indices,
                                 const ComputeType compute_type,
                                 const long max_queued_batches)
    : ReplicaPool(num_translators_per_device,
                  num_threads_per_translator,
                  model_dir,
                  device,
                  device_indices,
                  compute_type,
                  max_queued_batches)
  {
  }

  TranslatorPool::TranslatorPool(size_t num_translators_per_device,
                                 size_t num_threads_per_translator,
                                 models::ModelReader& model_reader,
                                 const Device device,
                                 const std::vector<int>& device_indices,
                                 const ComputeType compute_type,
                                 const long max_queued_batches)
    : ReplicaPool(num_translators_per_device,
                  num_threads_per_translator,
                  model_reader,
                  device,
                  device_indices,
                  compute_type,
                  max_queued_batches)
  {
  }

  std::vector<std::future<TranslationResult>>
  TranslatorPool::translate_batch_async(const std::vector<std::vector<std::string>>& source,
                                        const TranslationOptions& options,
                                        const size_t max_batch_size,
                                        const BatchType batch_type) {
    return translate_batch_async(source, {}, options, max_batch_size, batch_type);
  }

  std::vector<std::future<TranslationResult>>
  TranslatorPool::translate_batch_async(const std::vector<std::vector<std::string>>& source,
                                        const std::vector<std::vector<std::string>>& target_prefix,
                                        const TranslationOptions& options,
                                        const size_t max_batch_size,
                                        const BatchType batch_type) {
    return post_examples<TranslationResult>(
      load_examples({source, target_prefix}),
      max_batch_size,
      batch_type,
      [options](models::SequenceToSequenceReplica& model, const Batch& batch) {
        return run_translation(model, batch, options);
      });
  }

  std::vector<std::future<ScoringResult>>
  TranslatorPool::score_batch_async(const std::vector<std::vector<std::string>>& source,
                                    const std::vector<std::vector<std::string>>& target,
                                    const ScoringOptions& options,
                                    const size_t max_batch_size,
                                    const BatchType batch_type) {
    return post_examples<ScoringResult>(
      load_examples({source, target}),
      max_batch_size,
      batch_type,
      [options](models::SequenceToSequenceReplica& model, const Batch& batch) {
        return run_scoring(model, batch, options);
      });
  }

  std::vector<TranslationResult>
  TranslatorPool::translate_batch(const std::vector<std::vector<std::string>>& source,
                                  const TranslationOptions& options,
                                  const size_t max_batch_size,
                                  const BatchType batch_type) {
    return translate_batch(source, {}, options, max_batch_size, batch_type);
  }

  template <typename T>
  std::vector<T> get_results_from_futures(std::vector<std::future<T>> futures) {
    std::vector<T> results;
    results.reserve(futures.size());
    for (auto& future : futures)
      results.emplace_back(future.get());
    return results;
  }

  std::vector<TranslationResult>
  TranslatorPool::translate_batch(const std::vector<std::vector<std::string>>& source,
                                  const std::vector<std::vector<std::string>>& target_prefix,
                                  const TranslationOptions& options,
                                  const size_t max_batch_size,
                                  const BatchType batch_type) {
    return get_results_from_futures(translate_batch_async(source,
                                                          target_prefix,
                                                          options,
                                                          max_batch_size,
                                                          batch_type));
  }

  std::vector<ScoringResult>
  TranslatorPool::score_batch(const std::vector<std::vector<std::string>>& source,
                              const std::vector<std::vector<std::string>>& target,
                              const ScoringOptions& options,
                              const size_t max_batch_size,
                              const BatchType batch_type) {
    return get_results_from_futures(score_batch_async(source, target, options, max_batch_size, batch_type));
  }

  ExecutionStats TranslatorPool::translate_text_file(const std::string& source_file,
                                                     const std::string& output_file,
                                                     const TranslationOptions& options,
                                                     size_t max_batch_size,
                                                     size_t read_batch_size,
                                                     BatchType batch_type,
                                                     bool with_scores,
                                                     const std::string* target_file) {
    auto source = open_file<std::ifstream>(source_file);
    auto output = open_file<std::ofstream>(output_file);
    auto target = (target_file
                   ? std::make_unique<std::ifstream>(open_file<std::ifstream>(*target_file))
                   : nullptr);

    return translate_text_file(source,
                               output,
                               options,
                               max_batch_size,
                               read_batch_size,
                               batch_type,
                               with_scores,
                               target.get());
  }

  ExecutionStats TranslatorPool::translate_text_file(std::istream& source,
                                                     std::ostream& output,
                                                     const TranslationOptions& options,
                                                     size_t max_batch_size,
                                                     size_t read_batch_size,
                                                     BatchType batch_type,
                                                     bool with_scores,
                                                     std::istream* target) {
    return translate_raw_text_file(source,
                                   target,
                                   output,
                                   split_tokens,
                                   split_tokens,
                                   join_tokens,
                                   options,
                                   max_batch_size,
                                   read_batch_size,
                                   batch_type,
                                   with_scores);
  }

  ExecutionStats TranslatorPool::score_text_file(const std::string& source_file,
                                                 const std::string& target_file,
                                                 const std::string& output_file,
                                                 const ScoringOptions& options,
                                                 size_t max_batch_size,
                                                 size_t read_batch_size,
                                                 BatchType batch_type,
                                                 bool with_tokens_score) {
    auto source = open_file<std::ifstream>(source_file);
    auto target = open_file<std::ifstream>(target_file);
    auto output = open_file<std::ofstream>(output_file);
    return score_text_file(source,
                           target,
                           output,
                           options,
                           max_batch_size,
                           read_batch_size,
                           batch_type,
                           with_tokens_score);
  }

  ExecutionStats TranslatorPool::score_text_file(std::istream& source,
                                                 std::istream& target,
                                                 std::ostream& output,
                                                 const ScoringOptions& options,
                                                 size_t max_batch_size,
                                                 size_t read_batch_size,
                                                 BatchType batch_type,
                                                 bool with_tokens_score) {
    return score_raw_text_file(source,
                               target,
                               output,
                               split_tokens,
                               split_tokens,
                               join_tokens,
                               options,
                               max_batch_size,
                               read_batch_size,
                               batch_type,
                               with_tokens_score);
  }

  size_t TranslatorPool::num_translators() const {
    return ReplicaPool<models::SequenceToSequenceReplica>::num_replicas();
  }


  std::vector<ScoringResult>
  run_scoring(models::SequenceToSequenceReplica& model,
              const Batch& batch,
              const ScoringOptions& options) {
    spdlog::debug("Running batch scoring on {} examples", batch.num_examples());
    auto results = model.score(batch.get_stream(0), batch.get_stream(1), options);
    spdlog::debug("Finished batch scoring");
    return results;
  }

  std::vector<TranslationResult>
  run_translation(models::SequenceToSequenceReplica& model,
                  const Batch& batch,
                  const TranslationOptions& options) {
    spdlog::debug("Running batch translation on {} examples", batch.num_examples());
    auto results = model.translate(batch.get_stream(0), batch.get_stream(1), options);
    spdlog::debug("Finished batch translation");
    return results;
  }

}
