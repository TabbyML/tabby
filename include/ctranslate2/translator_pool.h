#pragma once

#include <chrono>
#include <future>
#include <fstream>
#include <mutex>
#include <queue>
#include <thread>

#include "batch_reader.h"
#include "translator.h"

namespace ctranslate2 {

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
                   const std::string& model_dir,
                   const Device device = Device::CPU,
                   const int device_index = 0,
                   const ComputeType compute_type = ComputeType::DEFAULT);

    // Multi-device constructor.
    TranslatorPool(size_t num_translators_per_device,
                   size_t num_threads_per_translator,
                   const std::string& model_dir,
                   const Device device,
                   const std::vector<int>& device_indices,
                   const ComputeType compute_type = ComputeType::DEFAULT);

    ~TranslatorPool();

    // Run a translation job asynchronously.
    std::future<std::vector<TranslationResult>>
    translate_batch_async(std::vector<std::vector<std::string>> source,
                          TranslationOptions options);
    std::future<std::vector<TranslationResult>>
    translate_batch_async(std::vector<std::vector<std::string>> source,
                          std::vector<std::vector<std::string>> target_prefix,
                          TranslationOptions options);

    // Run a translation synchronously.
    // To benefit from parallelism, you can set max_batch_size in the translation options:
    // the input will be split according to this value and each batch will be translated
    // in parallel.
    std::vector<TranslationResult>
    translate_batch(const std::vector<std::vector<std::string>>& source,
                    const TranslationOptions& options);
    std::vector<TranslationResult>
    translate_batch(const std::vector<std::vector<std::string>>& source,
                    const std::vector<std::vector<std::string>>& target_prefix,
                    const TranslationOptions& options);

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
      return consume_stream(in, nullptr, out, read_batch_size, options, reader, nullptr, writer);
    }

    template <typename SourceReader, typename TargetReader, typename TargetWriter>
    void consume_stream(std::istream& source,
                        std::istream* target,
                        std::ostream& output,
                        size_t read_batch_size,
                        const TranslationOptions& options,
                        SourceReader& source_reader,
                        TargetReader& target_reader,
                        TargetWriter& target_writer) {
      std::queue<std::future<std::vector<TranslationResult>>> results;

      auto pop_results = [&results, &output, &target_writer](bool blocking) {
        static const auto zero_sec = std::chrono::seconds(0);
        while (!results.empty()
               && (blocking
                   || results.front().wait_for(zero_sec) == std::future_status::ready)) {
          for (const auto& result : results.front().get())
            target_writer(output, result);
          results.pop();
        }
      };

      ParallelBatchReader batch_reader;
      batch_reader.add(std::make_unique<StreamReader<SourceReader>>(source, source_reader));
      if (target) {
        batch_reader.add(std::make_unique<StreamReader<TargetReader>>(*target, target_reader));
      }

      while (true) {
        auto batch = batch_reader.get_next(read_batch_size, options.batch_type);
        if (batch[0].empty())
          break;
        results.emplace(post(std::move(batch[0]),
                             target
                             ? std::move(batch[1])
                             : std::vector<std::vector<std::string>>(),
                             options,
                             /*throttle=*/true));

        pop_results(/*blocking=*/false);
      }

      pop_results(/*blocking=*/true);
    }

    // Translate a file in parallel.
    // These are wrappers around consume_stream that set the appropriate reader and writer.
    // The returned value is the total number of produced tokens.
    TranslationStats consume_text_file(const std::string& source_file,
                                       const std::string& output_file,
                                       size_t read_batch_size,
                                       const TranslationOptions& options,
                                       bool with_scores = false,
                                       const std::string* target_file = nullptr);

    TranslationStats consume_text_file(std::istream& source,
                                       std::ostream& output,
                                       size_t read_batch_size,
                                       const TranslationOptions& options,
                                       bool with_scores = false,
                                       std::istream* target = nullptr);

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
      return consume_raw_text_file(in,
                                   nullptr,
                                   out,
                                   tokenizer,
                                   tokenizer,
                                   detokenizer,
                                   read_batch_size,
                                   options,
                                   with_scores);
    }

    template <typename SourceTokenizer, typename TargetTokenizer, typename TargetDetokenizer>
    TranslationStats consume_raw_text_file(const std::string& source_file,
                                           const std::string* target_file,
                                           const std::string& output_file,
                                           SourceTokenizer& source_tokenizer,
                                           TargetTokenizer& target_tokenizer,
                                           TargetDetokenizer& detokenizer,
                                           const size_t read_batch_size,
                                           const TranslationOptions& options,
                                           const bool with_scores = false) {
      std::ifstream source;
      open_input_file(source_file, source);
      std::ofstream output;
      open_output_file(output_file, output);

      std::unique_ptr<std::ifstream> target;
      if (target_file) {
        target.reset(new std::ifstream());
        open_input_file(*target_file, *target);
      }

      return consume_raw_text_file(source,
                                   target.get(),
                                   output,
                                   source_tokenizer,
                                   target_tokenizer,
                                   detokenizer,
                                   read_batch_size,
                                   options,
                                   with_scores);
    }

    template <typename SourceTokenizer, typename TargetTokenizer, typename TargetDetokenizer>
    TranslationStats consume_raw_text_file(std::istream& source,
                                           std::istream* target,
                                           std::ostream& output,
                                           SourceTokenizer& source_tokenizer,
                                           TargetTokenizer& target_tokenizer,
                                           TargetDetokenizer& detokenizer,
                                           const size_t read_batch_size,
                                           const TranslationOptions& options,
                                           const bool with_scores = false) {
      TranslationStats stats;

      auto source_reader = [this, &source_tokenizer](std::istream& in,
                                                     std::vector<std::string>& tokens) {
        return read_next_sequence(in, source_tokenizer, tokens);
      };

      auto target_reader = [this, &target_tokenizer](std::istream& in,
                                                     std::vector<std::string>& tokens) {
        return read_next_sequence(in, target_tokenizer, tokens);
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

      consume_stream(source,
                     target,
                     output,
                     read_batch_size,
                     options,
                     source_reader,
                     target_reader,
                     writer);
      output.flush();

      const auto t2 = std::chrono::high_resolution_clock::now();
      stats.total_time_in_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
        t2 - t1).count();
      return stats;
    }

    size_t num_queued_batches();
    size_t num_translators() const;
    const std::vector<Translator>& get_translators() const;

    // With throttle=true it will block if there is already too much work pending.
    std::future<std::vector<TranslationResult>>
    post(std::vector<std::vector<std::string>> source,
         TranslationOptions options,
         bool throttle = false);
    std::future<std::vector<TranslationResult>>
    post(std::vector<std::vector<std::string>> source,
         std::vector<std::vector<std::string>> target_prefix,
         TranslationOptions options,
         bool throttle = false);

  private:
    class Job {
    public:
      virtual ~Job() = default;
      virtual void run(Translator& translator) = 0;
    };

    template <typename ResultType>
    class BaseJob : public Job {
    public:
      std::future<ResultType> get_future() {
        return _promise.get_future();
      }

      void run(Translator& translator) override;

    protected:
      virtual ResultType compute(Translator& translator) const = 0;

    private:
      std::promise<ResultType> _promise;
    };

    class TranslationJob : public BaseJob<std::vector<TranslationResult>> {
    public:
      TranslationJob(std::vector<std::vector<std::string>> source,
                     std::vector<std::vector<std::string>> target_prefix,
                     TranslationOptions options)
        : _source(std::move(source))
        , _target_prefix(std::move(target_prefix))
        , _options(std::move(options)) {
      }

    protected:
      std::vector<TranslationResult> compute(Translator& translator) const override;

    private:
      std::vector<std::vector<std::string>> _source;
      std::vector<std::vector<std::string>> _target_prefix;
      TranslationOptions _options;
    };

    void create_translators(size_t num_translators_per_device,
                            size_t num_threads_per_translator,
                            const std::string& model_dir,
                            const Device device,
                            std::vector<int> device_indices,
                            const ComputeType compute_type);

    void post_job(std::unique_ptr<Job> job, bool throttle = false);
    void work_loop(Translator& translator, size_t num_threads);

    void open_input_file(const std::string& file, std::ifstream& stream) const;
    void open_output_file(const std::string& file, std::ofstream& stream) const;

    std::condition_variable _can_add_more_work;
    std::queue<std::unique_ptr<Job>> _work;
    std::vector<std::thread> _workers;
    std::vector<Translator> _translators;
    std::mutex _mutex;
    std::condition_variable _cv;
    bool _request_end = false;

    template <typename Tokenizer>
    bool read_next_sequence(std::istream& in,
                            Tokenizer& tokenizer,
                            std::vector<std::string>& tokens) const {
      std::string line;
      if (!std::getline(in, line))
        return false;
      tokens = tokenizer(line);
      return true;
    }
  };

}
