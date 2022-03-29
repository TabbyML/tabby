#pragma once

#include <chrono>
#include <future>
#include <fstream>

#include "batch_reader.h"
#include "thread_pool.h"
#include "translator.h"

namespace ctranslate2 {

  struct TranslationStats {
    size_t num_tokens = 0;
    size_t num_examples = 0;
    double total_time_in_ms = 0;
  };

  // TranslatorPool is the high-level class for running translations. It supports parallel
  // and asynchronous translations.
  class TranslatorPool {
  public:
    // num_threads_per_translator (a.k.a. intra_threads) is forced to 1 when the translator
    // is running on a CUDA device.
    TranslatorPool(size_t num_translators,
                   size_t num_threads_per_translator,
                   const std::string& model_dir,
                   const Device device = Device::CPU,
                   const int device_index = 0,
                   const ComputeType compute_type = ComputeType::DEFAULT,
                   const long max_queued_batches = 0);

    // Constructor with ModelReader.
    TranslatorPool(size_t num_translators,
                   size_t num_threads_per_translator,
                   models::ModelReader& model_reader,
                   const Device device,
                   const int device_index,
                   const ComputeType compute_type = ComputeType::DEFAULT,
                   const long max_queued_batches = 0);

    // Multi-device constructor.
    TranslatorPool(size_t num_translators_per_device,
                   size_t num_threads_per_translator,
                   const std::string& model_dir,
                   const Device device,
                   const std::vector<int>& device_indices,
                   const ComputeType compute_type = ComputeType::DEFAULT,
                   const long max_queued_batches = 0);

    // Multi-device constructor with ModelReader.
    TranslatorPool(size_t num_translators_per_device,
                   size_t num_threads_per_translator,
                   models::ModelReader& model_reader,
                   const Device device,
                   const std::vector<int>& device_indices,
                   const ComputeType compute_type = ComputeType::DEFAULT,
                   const long max_queued_batches = 0);

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

    // Translate a stream.
    // The reader and writer functions do not need to be thread-safe.
    template <typename Reader, typename Writer>
    void consume_stream(std::istream& in,
                        std::ostream& out,
                        Reader& reader,
                        Writer& writer,
                        const TranslationOptions& options = TranslationOptions(),
                        size_t max_batch_size = 32,
                        size_t read_batch_size = 0,
                        BatchType batch_type = BatchType::Examples) {
      return consume_stream(in,
                            nullptr,
                            out,
                            reader,
                            nullptr,
                            writer,
                            options,
                            max_batch_size,
                            read_batch_size,
                            batch_type);
    }

    template <typename SourceReader, typename TargetReader, typename TargetWriter>
    void consume_stream(std::istream& source,
                        std::istream* target,
                        std::ostream& output,
                        SourceReader& source_reader,
                        TargetReader* target_reader,
                        TargetWriter& target_writer,
                        const TranslationOptions& options = TranslationOptions(),
                        size_t max_batch_size = 32,
                        size_t read_batch_size = 0,
                        BatchType batch_type = BatchType::Examples) {
      TranslateJobCreator job_creator(options);
      consume_stream(source,
                     target,
                     output,
                     source_reader,
                     target_reader,
                     target_writer,
                     job_creator,
                     max_batch_size,
                     read_batch_size,
                     batch_type);
    }

    // Translate a file.
    // These are wrappers around consume_stream that set the appropriate reader and writer.
    TranslationStats consume_text_file(const std::string& source_file,
                                       const std::string& output_file,
                                       const TranslationOptions& options = TranslationOptions(),
                                       size_t max_batch_size = 32,
                                       size_t read_batch_size = 0,
                                       BatchType batch_type = BatchType::Examples,
                                       bool with_scores = false,
                                       const std::string* target_file = nullptr);

    TranslationStats consume_text_file(std::istream& source,
                                       std::ostream& output,
                                       const TranslationOptions& options = TranslationOptions(),
                                       size_t max_batch_size = 32,
                                       size_t read_batch_size = 0,
                                       BatchType batch_type = BatchType::Examples,
                                       bool with_scores = false,
                                       std::istream* target = nullptr);

    template <typename Tokenizer, typename Detokenizer>
    TranslationStats consume_raw_text_file(const std::string& in_file,
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
      return consume_raw_text_file(in,
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
    TranslationStats consume_raw_text_file(std::istream& in,
                                           std::ostream& out,
                                           Tokenizer& tokenizer,
                                           Detokenizer& detokenizer,
                                           const TranslationOptions& options = TranslationOptions(),
                                           const size_t max_batch_size = 32,
                                           const size_t read_batch_size = 0,
                                           const BatchType batch_type = BatchType::Examples,
                                           const bool with_scores = false) {
      return consume_raw_text_file(in,
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
    TranslationStats consume_raw_text_file(const std::string& source_file,
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

      return consume_raw_text_file(source,
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
    TranslationStats consume_raw_text_file(std::istream& source,
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
      TranslationStats stats;

      TokensReader<SourceTokenizer> source_reader(source_tokenizer);
      TokensReader<TargetTokenizer> target_reader(target_tokenizer);

      auto writer = [&detokenizer, &stats, &with_scores](std::ostream& out,
                                                         const TranslationResult& result) {
        const auto& hypotheses = result.hypotheses;
        const auto& scores = result.scores;
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
                     source_reader,
                     &target_reader,
                     writer,
                     options,
                     max_batch_size,
                     read_batch_size,
                     batch_type);

      const auto t2 = std::chrono::high_resolution_clock::now();
      stats.total_time_in_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
        t2 - t1).count();
      return stats;
    }

    // Score a stream.
    // The reader and writer functions do not need to be thread-safe.
    template <typename SourceReader, typename TargetReader, typename TargetWriter>
    void score_stream(std::istream& source,
                      std::istream& target,
                      std::ostream& output,
                      SourceReader& source_reader,
                      TargetReader& target_reader,
                      TargetWriter& target_writer,
                      const ScoringOptions& options = ScoringOptions(),
                      size_t max_batch_size = 32,
                      size_t read_batch_size = 0,
                      BatchType batch_type = BatchType::Examples) {
      ScoreJobCreator job_creator(options);
      consume_stream(source,
                     &target,
                     output,
                     source_reader,
                     &target_reader,
                     target_writer,
                     job_creator,
                     max_batch_size,
                     read_batch_size,
                     batch_type);
    }

    // Score a file.
    // These are wrappers around score_stream that set the appropriate reader and writer.
    TranslationStats score_text_file(const std::string& source_file,
                                     const std::string& target_file,
                                     const std::string& output_file,
                                     const ScoringOptions& options = ScoringOptions(),
                                     size_t max_batch_size = 32,
                                     size_t read_batch_size = 0,
                                     BatchType batch_type = BatchType::Examples,
                                     bool with_tokens_score = false);
    TranslationStats score_text_file(std::istream& source,
                                     std::istream& target,
                                     std::ostream& output,
                                     const ScoringOptions& options = ScoringOptions(),
                                     size_t max_batch_size = 32,
                                     size_t read_batch_size = 0,
                                     BatchType batch_type = BatchType::Examples,
                                     bool with_tokens_score = false);

    template <typename SourceTokenizer, typename TargetTokenizer, typename TargetDetokenizer>
    TranslationStats score_raw_text_file(const std::string& source_file,
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
    TranslationStats score_raw_text_file(std::istream& source,
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
      TokensReader<SourceTokenizer> source_reader(source_tokenizer);
      TokensReader<TargetTokenizer> target_reader(target_tokenizer);
      TranslationStats stats;

      auto writer = [&target_detokenizer, &stats, with_token_scores](std::ostream& out,
                                                                     const ScoringResult& result) {
        stats.num_examples += 1;
        stats.num_tokens += result.tokens_score.size();
        out << result.normalized_score() << " ||| " << target_detokenizer(result.tokens);
        if (with_token_scores) {
          out << " |||";
          for (const auto score : result.tokens_score)
            out << ' ' << score;
        }
        out << '\n';
      };

      const auto t1 = std::chrono::high_resolution_clock::now();
      score_stream(source,
                   target,
                   output,
                   source_reader,
                   target_reader,
                   writer,
                   options,
                   max_batch_size,
                   read_batch_size,
                   batch_type);
      const auto t2 = std::chrono::high_resolution_clock::now();
      stats.total_time_in_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
        t2 - t1).count();
      return stats;
    }

    // Number of batches in the work queue.
    size_t num_queued_batches();
    // Number of batches in the work queue or currently processed by a worker.
    size_t num_active_batches() const;

    size_t num_translators() const;
    const Translator& get_translator(size_t index) const;

    // Return the translator local to the current thread.
    static Translator* get_translator();

    void clear_cache() const;

  private:
    friend class BufferedTranslationWrapper;

    // Base class for consuming job results.
    template <typename Result>
    class JobResultConsumer {
    public:
      JobResultConsumer(size_t num_results)
        : _promises(num_results)
      {
      }

      JobResultConsumer(std::vector<std::promise<Result>> promises)
        : _promises(std::move(promises))
      {
      }

      std::vector<std::future<Result>> get_futures() {
        std::vector<std::future<Result>> futures;
        futures.reserve(_promises.size());
        for (auto& promise : _promises)
          futures.emplace_back(promise.get_future());
        return futures;
      }

      void set_result(size_t index, Result result) {
        _promises[index].set_value(std::move(result));
      }

      void set_exception(size_t index, std::exception_ptr exception) {
        _promises[index].set_exception(exception);
      }

    private:
      std::vector<std::promise<Result>> _promises;
    };

    template <typename Result>
    class BatchJob : public Job {
    public:
      BatchJob(Batch batch, std::shared_ptr<JobResultConsumer<Result>> consumer)
        : _batch(std::move(batch))
        , _consumer(std::move(consumer))
      {
      }

      void run() override {
        std::vector<Result> results;
        std::exception_ptr exception;

        try {
          results = get_results();
        } catch (...) {
          exception = std::current_exception();
        }

        for (size_t i = 0; i < _batch.num_examples(); ++i) {
          const size_t index = (_batch.example_index.empty() ? i : _batch.example_index[i]);
          if (exception)
            _consumer->set_exception(index, exception);
          else
            _consumer->set_result(index, std::move(results[i]));
        }
      }

    protected:
      virtual std::vector<Result> get_results() const = 0;
      const Batch _batch;

    private:
      const std::shared_ptr<JobResultConsumer<Result>> _consumer;
    };

    class TranslateJob : public BatchJob<TranslationResult> {
    public:
      TranslateJob(Batch batch,
                   TranslationOptions options,
                   std::shared_ptr<JobResultConsumer<TranslationResult>> consumer);

    protected:
      std::vector<TranslationResult> get_results() const override;

    private:
      const TranslationOptions _options;
    };

    class ScoreJob : public BatchJob<ScoringResult> {
    public:
      ScoreJob(Batch batch,
               ScoringOptions options,
               std::shared_ptr<JobResultConsumer<ScoringResult>> consumer);

    protected:
      std::vector<ScoringResult> get_results() const override;

    private:
      const ScoringOptions _options;
    };

    template <typename Result>
    class JobCreator {
    public:
      virtual ~JobCreator() = default;

      std::vector<std::future<Result>> post(ThreadPool& pool,
                                            const std::vector<Example>& examples,
                                            size_t max_batch_size,
                                            BatchType batch_type) const {
        if (examples.empty())
          return {};

        auto batches = rebatch_input(examples, max_batch_size, batch_type);
        auto consumer = std::make_shared<JobResultConsumer<Result>>(examples.size());
        auto futures = consumer->get_futures();

        for (auto& batch : batches)
          pool.post(create_job(std::move(batch), consumer));

        return futures;
      }

    protected:
      virtual std::unique_ptr<Job>
      create_job(Batch batch, std::shared_ptr<JobResultConsumer<Result>> consumer) const = 0;
    };

    class TranslateJobCreator : public JobCreator<TranslationResult> {
    public:
      TranslateJobCreator(TranslationOptions options)
        : _options(std::move(options))
      {
        _options.validate();
      }

    protected:
      std::unique_ptr<Job>
      create_job(Batch batch,
                 std::shared_ptr<JobResultConsumer<TranslationResult>> consumer) const override {
        return std::make_unique<TranslateJob>(std::move(batch), _options, std::move(consumer));
      }

    private:
      const TranslationOptions _options;
    };

    class ScoreJobCreator : public JobCreator<ScoringResult> {
    public:
      ScoreJobCreator(ScoringOptions options)
        : _options(std::move(options))
      {
      }
    protected:
      std::unique_ptr<Job>
      create_job(Batch batch,
                 std::shared_ptr<JobResultConsumer<ScoringResult>> consumer) const override {
        return std::make_unique<ScoreJob>(std::move(batch), _options, std::move(consumer));
      }
    private:
      const ScoringOptions _options;
    };

    template <typename SourceReader,
              typename TargetReader,
              typename TargetWriter,
              typename Result>
    void consume_stream(std::istream& source,
                        std::istream* target,
                        std::ostream& output,
                        SourceReader& source_reader,
                        TargetReader* target_reader,
                        TargetWriter& target_writer,
                        const JobCreator<Result>& job_creator,
                        size_t max_batch_size,
                        size_t read_batch_size,
                        BatchType batch_type) {
      std::queue<std::future<Result>> results;

      auto pop_results = [&results, &output, &target_writer](bool blocking) {
        constexpr std::chrono::seconds zero_sec(0);
        while (!results.empty()
               && (blocking
                   || results.front().wait_for(zero_sec) == std::future_status::ready)) {
          target_writer(output, results.front().get());
          results.pop();
        }
      };

      std::unique_ptr<BatchReader> batch_reader;
      if (target) {
        auto parallel_reader = std::make_unique<ParallelBatchReader>();
        parallel_reader->add(std::make_unique<StreamReader<SourceReader>>(source, source_reader));
        parallel_reader->add(std::make_unique<StreamReader<TargetReader>>(*target, *target_reader));
        batch_reader = std::move(parallel_reader);
      } else {
        batch_reader = std::make_unique<StreamReader<SourceReader>>(source, source_reader);
      }

      if (read_batch_size == 0)
        read_batch_size = (max_batch_size == 1 ? max_batch_size : max_batch_size * 16);

      while (true) {
        auto examples = batch_reader->get_next(read_batch_size, batch_type);
        if (examples.empty())
          break;
        auto futures = job_creator.post(*_thread_pool,
                                        examples,
                                        max_batch_size,
                                        batch_type);
        for (auto& future : futures)
          results.emplace(std::move(future));

        pop_results(/*blocking=*/false);
      }

      pop_results(/*blocking=*/true);
      output.flush();
    }

    class TranslatorWorker : public Worker {
    public:
      TranslatorWorker(const std::shared_ptr<const models::Model>& model, size_t num_threads);

      Translator& translator() {
        return _translator;
      }

      Allocator* allocator() {
        return _allocator;
      }

    protected:
      void initialize() override;
      void finalize() override;

    private:
      Translator _translator;
      Allocator* _allocator;
      const Device _device;
      const size_t _num_threads;
    };

    TranslatorWorker& get_worker(size_t index) const;

    void create_translators(size_t num_translators_per_device,
                            size_t num_threads_per_translator,
                            models::ModelReader& model_reader,
                            const Device device,
                            std::vector<int> device_indices,
                            const ComputeType compute_type,
                            const long max_queued_batches);

    std::unique_ptr<ThreadPool> _thread_pool;

    template <typename Tokenizer>
    class TokensReader {
    public:
      TokensReader(Tokenizer& tokenizer)
        : _tokenizer(tokenizer)
      {
      }

      bool operator()(std::istream& in, std::vector<std::string>& tokens) {
        std::string line;
        if (!std::getline(in, line))
          return false;
        tokens = _tokenizer(line);
        return true;
      }

    private:
      Tokenizer& _tokenizer;
    };
  };

}
