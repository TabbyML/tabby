#pragma once

#include <fstream>

#include "replica_pool.h"
#include "models/sequence_to_sequence.h"

namespace ctranslate2 {

  struct ExecutionStats {
    size_t num_tokens = 0;
    size_t num_examples = 0;
    double total_time_in_ms = 0;
  };

  std::vector<ScoringResult>
  run_scoring(models::SequenceToSequenceReplica& translator,
              const Batch& batch,
              const ScoringOptions& options);
  std::vector<TranslationResult>
  run_translation(models::SequenceToSequenceReplica& translator,
                  const Batch& batch,
                  const TranslationOptions& options);

  // Translator is the high-level class for running translations. It supports parallel
  // and asynchronous translations.
  class Translator : public ReplicaPool<models::SequenceToSequenceReplica> {
  public:
    using ReplicaPool::ReplicaPool;

    std::vector<std::future<TranslationResult>>
    translate_batch_async(const std::vector<std::vector<std::string>>& source,
                          const TranslationOptions& options = TranslationOptions(),
                          const size_t max_batch_size = 0,
                          const BatchType batch_type = BatchType::Examples);
    std::vector<std::future<TranslationResult>>
    translate_batch_async(const std::vector<std::vector<std::string>>& source,
                          const std::vector<std::vector<std::string>>& target_prefix,
                          const TranslationOptions& options = TranslationOptions(),
                          const size_t max_batch_size = 0,
                          const BatchType batch_type = BatchType::Examples);

    std::vector<TranslationResult>
    translate_batch(const std::vector<std::vector<std::string>>& source,
                    const TranslationOptions& options = TranslationOptions(),
                    const size_t max_batch_size = 0,
                    const BatchType batch_type = BatchType::Examples);
    std::vector<TranslationResult>
    translate_batch(const std::vector<std::vector<std::string>>& source,
                    const std::vector<std::vector<std::string>>& target_prefix,
                    const TranslationOptions& options = TranslationOptions(),
                    const size_t max_batch_size = 0,
                    const BatchType batch_type = BatchType::Examples);

    std::vector<std::future<ScoringResult>>
    score_batch_async(const std::vector<std::vector<std::string>>& source,
                      const std::vector<std::vector<std::string>>& target,
                      const ScoringOptions& options = ScoringOptions(),
                      const size_t max_batch_size = 0,
                      const BatchType batch_type = BatchType::Examples);
    std::vector<ScoringResult>
    score_batch(const std::vector<std::vector<std::string>>& source,
                const std::vector<std::vector<std::string>>& target,
                const ScoringOptions& options = ScoringOptions(),
                const size_t max_batch_size = 0,
                const BatchType batch_type = BatchType::Examples);

    // Translate a file.
    ExecutionStats translate_text_file(const std::string& source_file,
                                       const std::string& output_file,
                                       const TranslationOptions& options = TranslationOptions(),
                                       size_t max_batch_size = 32,
                                       size_t read_batch_size = 0,
                                       BatchType batch_type = BatchType::Examples,
                                       bool with_scores = false,
                                       const std::string* target_file = nullptr);

    ExecutionStats translate_text_file(std::istream& source,
                                       std::ostream& output,
                                       const TranslationOptions& options = TranslationOptions(),
                                       size_t max_batch_size = 32,
                                       size_t read_batch_size = 0,
                                       BatchType batch_type = BatchType::Examples,
                                       bool with_scores = false,
                                       std::istream* target = nullptr);

    template <typename Tokenizer, typename Detokenizer>
    ExecutionStats translate_raw_text_file(const std::string& in_file,
                                           const std::string& out_file,
                                           Tokenizer& tokenizer,
                                           Detokenizer& detokenizer,
                                           const TranslationOptions& options = TranslationOptions(),
                                           const size_t max_batch_size = 32,
                                           const size_t read_batch_size = 0,
                                           const BatchType batch_type = BatchType::Examples,
                                           const bool with_scores = false) {
      auto in = open_file<std::ifstream>(in_file);
      auto out = open_file<std::ofstream>(out_file);
      return translate_raw_text_file(in,
                                     out,
                                     tokenizer,
                                     detokenizer,
                                     options,
                                     max_batch_size,
                                     read_batch_size,
                                     batch_type,
                                     with_scores);

    }

    template <typename Tokenizer, typename Detokenizer>
    ExecutionStats translate_raw_text_file(std::istream& in,
                                           std::ostream& out,
                                           Tokenizer& tokenizer,
                                           Detokenizer& detokenizer,
                                           const TranslationOptions& options = TranslationOptions(),
                                           const size_t max_batch_size = 32,
                                           const size_t read_batch_size = 0,
                                           const BatchType batch_type = BatchType::Examples,
                                           const bool with_scores = false) {
      return translate_raw_text_file(in,
                                     nullptr,
                                     out,
                                     tokenizer,
                                     tokenizer,
                                     detokenizer,
                                     options,
                                     max_batch_size,
                                     read_batch_size,
                                     batch_type,
                                     with_scores);
    }

    template <typename SourceTokenizer, typename TargetTokenizer, typename TargetDetokenizer>
    ExecutionStats translate_raw_text_file(const std::string& source_file,
                                           const std::string* target_file,
                                           const std::string& output_file,
                                           SourceTokenizer& source_tokenizer,
                                           TargetTokenizer& target_tokenizer,
                                           TargetDetokenizer& detokenizer,
                                           const TranslationOptions& options = TranslationOptions(),
                                           const size_t max_batch_size = 32,
                                           const size_t read_batch_size = 0,
                                           const BatchType batch_type = BatchType::Examples,
                                           const bool with_scores = false) {
      auto source = open_file<std::ifstream>(source_file);
      auto output = open_file<std::ofstream>(output_file);
      auto target = (target_file
                     ? std::make_unique<std::ifstream>(open_file<std::ifstream>(*target_file))
                     : nullptr);

      return translate_raw_text_file(source,
                                     target.get(),
                                     output,
                                     source_tokenizer,
                                     target_tokenizer,
                                     detokenizer,
                                     options,
                                     max_batch_size,
                                     read_batch_size,
                                     batch_type,
                                     with_scores);
    }

    template <typename SourceTokenizer, typename TargetTokenizer, typename TargetDetokenizer>
    ExecutionStats translate_raw_text_file(std::istream& source,
                                           std::istream* target,
                                           std::ostream& output,
                                           SourceTokenizer& source_tokenizer,
                                           TargetTokenizer& target_tokenizer,
                                           TargetDetokenizer& detokenizer,
                                           const TranslationOptions& options = TranslationOptions(),
                                           const size_t max_batch_size = 32,
                                           const size_t read_batch_size = 0,
                                           const BatchType batch_type = BatchType::Examples,
                                           const bool with_scores = false) {
      ExecutionStats stats;

      TextLineReader<SourceTokenizer> source_reader(source_tokenizer);
      TextLineReader<TargetTokenizer> target_reader(target_tokenizer);

      auto writer = [&detokenizer, &stats, &output, &with_scores](const TranslationResult& result) {
        const auto& hypotheses = result.hypotheses;
        const auto& scores = result.scores;
        stats.num_examples += 1;
        stats.num_tokens += hypotheses[0].size();
        for (size_t n = 0; n < hypotheses.size(); ++n) {
          if (with_scores)
            output << (result.has_scores() ? scores[n] : 0) << " ||| ";
          output << detokenizer(hypotheses[n]) << '\n';
        }
      };

      const auto t1 = std::chrono::high_resolution_clock::now();

      consume_stream<TranslationResult>(
        source,
        target,
        output,
        source_reader,
        &target_reader,
        writer,
        max_batch_size,
        read_batch_size,
        batch_type,
        [options](models::SequenceToSequenceReplica& model, const Batch& batch) {
          return run_translation(model, batch, options);
        });

      const auto t2 = std::chrono::high_resolution_clock::now();
      stats.total_time_in_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
        t2 - t1).count();
      return stats;
    }

    // Score a file.
    ExecutionStats score_text_file(const std::string& source_file,
                                   const std::string& target_file,
                                   const std::string& output_file,
                                   const ScoringOptions& options = ScoringOptions(),
                                   size_t max_batch_size = 32,
                                   size_t read_batch_size = 0,
                                   BatchType batch_type = BatchType::Examples,
                                   bool with_tokens_score = false);
    ExecutionStats score_text_file(std::istream& source,
                                   std::istream& target,
                                   std::ostream& output,
                                   const ScoringOptions& options = ScoringOptions(),
                                   size_t max_batch_size = 32,
                                   size_t read_batch_size = 0,
                                   BatchType batch_type = BatchType::Examples,
                                   bool with_tokens_score = false);

    template <typename SourceTokenizer, typename TargetTokenizer, typename TargetDetokenizer>
    ExecutionStats score_raw_text_file(const std::string& source_file,
                                       const std::string& target_file,
                                       const std::string& output_file,
                                       SourceTokenizer& source_tokenizer,
                                       TargetTokenizer& target_tokenizer,
                                       TargetDetokenizer& target_detokenizer,
                                       const ScoringOptions& options = ScoringOptions(),
                                       const size_t max_batch_size = 32,
                                       const size_t read_batch_size = 0,
                                       const BatchType batch_type = BatchType::Examples,
                                       bool with_tokens_score = false) {
      auto source = open_file<std::ifstream>(source_file);
      auto target = open_file<std::ifstream>(target_file);
      auto output = open_file<std::ofstream>(output_file);
      return score_raw_text_file(source,
                                 target,
                                 output,
                                 source_tokenizer,
                                 target_tokenizer,
                                 target_detokenizer,
                                 options,
                                 max_batch_size,
                                 read_batch_size,
                                 batch_type,
                                 with_tokens_score);
    }

    template <typename SourceTokenizer, typename TargetTokenizer, typename TargetDetokenizer>
    ExecutionStats score_raw_text_file(std::istream& source,
                                       std::istream& target,
                                       std::ostream& output,
                                       SourceTokenizer& source_tokenizer,
                                       TargetTokenizer& target_tokenizer,
                                       TargetDetokenizer& target_detokenizer,
                                       const ScoringOptions& options = ScoringOptions(),
                                       const size_t max_batch_size = 32,
                                       const size_t read_batch_size = 0,
                                       const BatchType batch_type = BatchType::Examples,
                                       bool with_token_scores = false) {
      TextLineReader<SourceTokenizer> source_reader(source_tokenizer);
      TextLineReader<TargetTokenizer> target_reader(target_tokenizer);
      ExecutionStats stats;

      auto writer = [&target_detokenizer, &stats, &output, with_token_scores](const ScoringResult& result) {
        stats.num_examples += 1;
        stats.num_tokens += result.tokens_score.size();
        output << result.normalized_score() << " ||| " << target_detokenizer(result.tokens);
        if (with_token_scores) {
          output << " |||";
          for (const auto score : result.tokens_score)
            output << ' ' << score;
        }
        output << '\n';
      };

      const auto t1 = std::chrono::high_resolution_clock::now();

      consume_stream<ScoringResult>(
        source,
        &target,
        output,
        source_reader,
        &target_reader,
        writer,
        max_batch_size,
        read_batch_size,
        batch_type,
        [options](models::SequenceToSequenceReplica& model, const Batch& batch) {
          return run_scoring(model, batch, options);
        });

      const auto t2 = std::chrono::high_resolution_clock::now();
      stats.total_time_in_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
        t2 - t1).count();
      return stats;
    }

  private:
    friend class BufferedTranslationWrapper;

    template <typename Result,
              typename SourceReader,
              typename TargetReader,
              typename TargetWriter,
              typename Func>
    void consume_stream(std::istream& source,
                        std::istream* target,
                        std::ostream& output,
                        SourceReader& source_reader,
                        TargetReader* target_reader,
                        TargetWriter& target_writer,
                        size_t max_batch_size,
                        size_t read_batch_size,
                        BatchType batch_type,
                        const Func& func) {
      std::unique_ptr<BatchReader> batch_reader;
      if (target) {
        auto parallel_reader = std::make_unique<ParallelBatchReader>();
        parallel_reader->add(std::make_unique<StreamReader<SourceReader>>(source, source_reader));
        parallel_reader->add(std::make_unique<StreamReader<TargetReader>>(*target, *target_reader));
        batch_reader = std::move(parallel_reader);
      } else {
        batch_reader = std::make_unique<StreamReader<SourceReader>>(source, source_reader);
      }

      consume_batches<Result>(*batch_reader,
                              target_writer,
                              func,
                              max_batch_size,
                              read_batch_size,
                              batch_type);

      output.flush();
    }
  };

}
