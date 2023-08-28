#include "ctranslate2/translator.h"

#include <spdlog/spdlog.h>

namespace ctranslate2 {

  std::vector<std::future<TranslationResult>>
  Translator::translate_batch_async(const std::vector<std::vector<std::string>>& source,
                                    const TranslationOptions& options,
                                    const size_t max_batch_size,
                                    const BatchType batch_type) {
    return translate_batch_async(source, {}, options, max_batch_size, batch_type);
  }

  std::vector<std::future<TranslationResult>>
  Translator::translate_batch_async(const std::vector<std::vector<std::string>>& source,
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
  Translator::score_batch_async(const std::vector<std::vector<std::string>>& source,
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
  Translator::translate_batch(const std::vector<std::vector<std::string>>& source,
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
  Translator::translate_batch(const std::vector<std::vector<std::string>>& source,
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
  Translator::score_batch(const std::vector<std::vector<std::string>>& source,
                          const std::vector<std::vector<std::string>>& target,
                          const ScoringOptions& options,
                          const size_t max_batch_size,
                          const BatchType batch_type) {
    return get_results_from_futures(score_batch_async(source, target, options, max_batch_size, batch_type));
  }

  ExecutionStats Translator::translate_text_file(const std::string& source_file,
                                                 const std::string& output_file,
                                                 const TranslationOptions& options,
                                                 size_t max_batch_size,
                                                 size_t read_batch_size,
                                                 BatchType batch_type,
                                                 bool with_scores,
                                                 const std::string* target_file) {
    auto source = open_file_read(source_file);
    auto output = open_file_write(output_file);
    auto target = (target_file
                   ? std::make_unique<std::ifstream>(open_file_read(*target_file))
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

  ExecutionStats Translator::translate_text_file(std::istream& source,
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

  ExecutionStats Translator::score_text_file(const std::string& source_file,
                                             const std::string& target_file,
                                             const std::string& output_file,
                                             const ScoringOptions& options,
                                             size_t max_batch_size,
                                             size_t read_batch_size,
                                             BatchType batch_type,
                                             bool with_tokens_score) {
    auto source = open_file_read(source_file);
    auto target = open_file_read(target_file);
    auto output = open_file_write(output_file);
    return score_text_file(source,
                           target,
                           output,
                           options,
                           max_batch_size,
                           read_batch_size,
                           batch_type,
                           with_tokens_score);
  }

  ExecutionStats Translator::score_text_file(std::istream& source,
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
    auto results = model.translate(batch.get_stream(0),
                                   batch.get_stream(1),
                                   restore_batch_ids_in_callback(options, batch.example_index));
    spdlog::debug("Finished batch translation");
    return results;
  }

}
