#pragma once

#include <chrono>
#include <future>
#include <fstream>
#include <mutex>
#include <queue>
#include <thread>

#include "translator.h"

namespace ctranslate2 {

  using TranslationInput = std::vector<std::vector<std::string>>;
  using TranslationOutput = std::vector<TranslationResult>;

  struct TranslationStats {
    size_t num_tokens = 0;
    size_t num_examples = 0;
    double total_time_in_ms = 0;
  };

  // A pool of Translators running in parallel.
  class TranslatorPool {
  public:
    TranslatorPool(size_t num_translators,
                   size_t num_threads_per_translator,
                   const std::shared_ptr<const models::Model>& model);

    // "args" are forwarded to the models::Model::load function.
    template <typename... Args>
    TranslatorPool(size_t num_translators,
                   size_t num_threads_per_translator,
                   const std::string& model_dir,
                   Args&&... args) {
      const auto model = models::Model::load(model_dir, std::forward<Args>(args)...);
      create_translators(model, num_translators, num_threads_per_translator);
    }

    ~TranslatorPool();

    // Run a translation job asynchronously.
    // With blocking=true it will block if there is already too much work pending.
    std::future<TranslationOutput> post(TranslationInput source,
                                        TranslationOptions options,
                                        bool blocking=false);
    std::future<TranslationOutput> post(TranslationInput source,
                                        TranslationInput target_prefix,
                                        TranslationOptions options,
                                        bool blocking=false);

    // Translate a stream in parallel.
    // Results will be written in order as they are available so the stream content is
    // never stored fully in memory
    // The reader and writer functions do not need to be thread-safe .
    template <typename Reader, typename Writer>
    void consume_stream(std::istream& in,
                        std::ostream& out,
                        size_t read_batch_size,
                        const TranslationOptions& options,
                        Reader& reader,
                        Writer& writer) {
      std::queue<std::future<TranslationOutput>> results;

      auto pop_results = [&results, &out, &writer](bool blocking) {
        static const auto zero_sec = std::chrono::seconds(0);
        while (!results.empty()
               && (blocking
                   || results.front().wait_for(zero_sec) == std::future_status::ready)) {
          for (const auto& result : results.front().get())
            writer(out, result);
          results.pop();
        }
      };

      TranslationInput batch_tokens;
      batch_tokens.reserve(read_batch_size);
      std::vector<std::string> tokens;
      size_t batch_size = 0;

      while (reader(in, tokens)) {
        const size_t batch_size_increment = get_batch_size_increment(tokens, options.batch_type);

        if (batch_size > 0 && batch_size + batch_size_increment > read_batch_size) {
          results.emplace(post(std::move(batch_tokens), options, true));
          batch_tokens.clear();
          batch_size = 0;
          pop_results(false /* blocking */);
        }

        batch_tokens.emplace_back(std::move(tokens));
        batch_size += batch_size_increment;
        tokens.clear();
      }

      if (!batch_tokens.empty())
        results.emplace(post(std::move(batch_tokens), options, true));

      pop_results(true /* blocking */);
    }

    // Translate a file in parallel.
    // These are wrappers around consume_stream that set the appropriate reader and writer.
    // The returned value is the total number of produced tokens.
    TranslationStats consume_text_file(const std::string& in_file,
                                       const std::string& out_file,
                                       size_t read_batch_size,
                                       const TranslationOptions& options,
                                       bool with_scores = false);

    TranslationStats consume_text_file(std::istream& in,
                                       std::ostream& out,
                                       size_t read_batch_size,
                                       const TranslationOptions& options,
                                       bool with_scores = false);

    template <typename Tokenizer, typename Detokenizer>
    TranslationStats consume_raw_text_file(const std::string& in_file,
                                           const std::string& out_file,
                                           Tokenizer& tokenizer,
                                           Detokenizer& detokenizer,
                                           const size_t read_batch_size,
                                           const TranslationOptions& options,
                                           const bool with_scores = false) {
      std::ifstream in;
      open_input_file(in_file, in);
      std::ofstream out;
      open_output_file(out_file, out);
      return consume_raw_text_file(in,
                                   out,
                                   tokenizer,
                                   detokenizer,
                                   read_batch_size,
                                   options,
                                   with_scores);

    }

    template <typename Tokenizer, typename Detokenizer>
    TranslationStats consume_raw_text_file(std::istream& in,
                                           std::ostream& out,
                                           Tokenizer& tokenizer,
                                           Detokenizer& detokenizer,
                                           const size_t read_batch_size,
                                           const TranslationOptions& options,
                                           const bool with_scores = false) {
      TranslationStats stats;

      auto reader = [&tokenizer](std::istream& in, std::vector<std::string>& tokens) {
        std::string line;
        if (!std::getline(in, line))
          return false;
        tokens = tokenizer(line);
        return true;
      };

      auto writer = [&detokenizer, &stats, &with_scores](std::ostream& out,
                                                         const TranslationResult& result) {
        const auto& hypotheses = result.hypotheses();
        const auto& scores = result.scores();
        stats.num_examples += 1;
        stats.num_tokens += hypotheses[0].size();
        for (size_t n = 0; n < hypotheses.size(); ++n) {
          if (with_scores)
            out << (result.has_scores() ? scores[n] : 0) << " ||| ";
          out << detokenizer(hypotheses[n]) << '\n';
        }
      };

      const auto t1 = std::chrono::high_resolution_clock::now();

      consume_stream(in, out, read_batch_size, options, reader, writer);
      out.flush();

      const auto t2 = std::chrono::high_resolution_clock::now();
      stats.total_time_in_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
        t2 - t1).count();
      return stats;
    }

    size_t num_queued_batches();
    size_t num_translators() const;
    const std::vector<Translator>& get_translators() const;

  private:
    struct TranslationJob {
      TranslationJob(TranslationInput source_,
                     TranslationInput target_prefix_,
                     TranslationOptions options_)
        : source(source_)
        , target_prefix(target_prefix_)
        , options(options_) {
      }
      const TranslationInput source;
      const TranslationInput target_prefix;
      const TranslationOptions options;
    };

    void create_translators(const std::shared_ptr<const models::Model>& model,
                            size_t num_translators,
                            size_t num_threads_per_translator);
    void work_loop(Translator& translator, size_t num_threads);

    void open_input_file(const std::string& file, std::ifstream& stream) const;
    void open_output_file(const std::string& file, std::ofstream& stream) const;

    std::condition_variable _can_add_more_work;
    std::queue<std::pair<const TranslationJob, std::promise<TranslationOutput>>> _work;
    std::vector<std::thread> _workers;
    std::vector<Translator> _translators;
    std::mutex _mutex;
    std::condition_variable _cv;
    bool _request_end = false;
  };

}
