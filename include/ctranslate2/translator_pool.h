#pragma once

#include <future>
#include <istream>
#include <mutex>
#include <ostream>
#include <queue>
#include <thread>

#include "translator.h"

namespace ctranslate2 {

  using TranslationInput = std::vector<std::vector<std::string>>;
  using TranslationOutput = std::vector<TranslationResult>;

  // A pool of Translators running in parallel.
  class TranslatorPool {
  public:
    // "args" are forwarded to the Translator constructor.
    template <typename... Args>
    TranslatorPool(size_t num_replicas, size_t num_threads_per_replica, Args&&... args) {
      set_num_threads(num_threads_per_replica);
      _translator_pool.emplace_back(std::forward<Args>(args)...);
      // On GPU, we currently don't benefit much from running instances in parallel, even
      // when using separate streams. This could be revisited/improved in the future.
      if (_translator_pool.back().device() == Device::CUDA)
        num_replicas = 1;
      for (size_t i = 1; i < num_replicas; ++i)
        _translator_pool.emplace_back(_translator_pool.front());
      for (auto& translator : _translator_pool)
        _workers.emplace_back(&TranslatorPool::work_loop,
                              this,
                              std::ref(translator),
                              num_threads_per_replica);
    }

    ~TranslatorPool();

    // Run a translation job asynchronously.
    // With blocking=true it will block if there is already too much work pending.
    std::future<TranslationOutput> post(const TranslationInput& source,
                                        const TranslationOptions& options,
                                        bool blocking=false);
    std::future<TranslationOutput> post(const TranslationInput& source,
                                        const TranslationInput& target_prefix,
                                        const TranslationOptions& options,
                                        bool blocking=false);

    // Translate a stream in parallel.
    // Results will be written in order as they are available so the stream content is
    // never stored fully in memory.
    template <typename Reader, typename Writer>
    void consume_stream(std::istream& in,
                        std::ostream& out,
                        size_t max_batch_size,
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
      std::vector<std::string> tokens;

      while (reader(in, tokens)) {
        batch_tokens.push_back(tokens);
        tokens.clear();
        if (batch_tokens.size() == max_batch_size) {
          results.emplace(post(batch_tokens, options, true));
          batch_tokens.clear();
        }
        pop_results(false /* blocking */);
      }

      if (!batch_tokens.empty())
        results.emplace(post(batch_tokens, options, true));

      pop_results(true /* blocking */);
    }

    // Translate a file in parallel.
    // These are wrappers around consume_stream that set the appropriate reader and writer.
    // The returned value is the total number of produced tokens.
    size_t consume_text_file(const std::string& in_file,
                             const std::string& out_file,
                             size_t max_batch_size,
                             const TranslationOptions& options,
                             bool with_scores = false);

    size_t consume_text_file(std::istream& in,
                             std::ostream& out,
                             size_t max_batch_size,
                             const TranslationOptions& options,
                             bool with_scores = false);

  private:
    struct TranslationJob {
      TranslationInput source;
      TranslationInput target_prefix;
      TranslationOptions options;
    };

    void work_loop(Translator& translator, size_t intra_threads);

    std::condition_variable _can_add_more_work;
    std::queue<std::pair<TranslationJob, std::promise<TranslationOutput>>> _work;
    std::vector<std::thread> _workers;
    std::vector<Translator> _translator_pool;
    std::mutex _mutex;
    std::condition_variable _cv;
    bool _request_end = false;
  };

}
