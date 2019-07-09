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

  struct TranslationJob {
    TranslationInput source;
    TranslationInput target_prefix;
    TranslationOptions options;
  };

  class TranslatorPool {
  public:
    template <typename... Args>
    TranslatorPool(size_t num_replicas, size_t num_threads_per_replica, Args&&... args) {
      _translator_pool.emplace_back(std::forward<Args>(args)...);
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

    std::future<TranslationOutput> post(const TranslationInput& source,
                                        const TranslationOptions& options);
    std::future<TranslationOutput> post(const TranslationInput& source,
                                        const TranslationInput& target_prefix,
                                        const TranslationOptions& options);

    template <typename Reader, typename Writer>
    void consume_stream(std::istream& in,
                        std::ostream& out,
                        size_t max_batch_size,
                        const TranslationOptions& options,
                        Reader& reader,
                        Writer& writer) {
      std::queue<std::future<TranslationOutput>> futures;

      auto pop_results = [&futures, &out, &writer](bool blocking) {
        static const auto zero_sec = std::chrono::seconds(0);
        while (!futures.empty()
               && (blocking
                   || futures.front().wait_for(zero_sec) == std::future_status::ready)) {
          for (const auto& result : futures.front().get())
            writer(out, result);
          futures.pop();
        }
      };

      TranslationInput batch_tokens;
      std::vector<std::string> tokens;

      while (reader(in, tokens)) {
        batch_tokens.push_back(tokens);
        tokens.clear();
        if (batch_tokens.size() == max_batch_size) {
          futures.emplace(post(batch_tokens, options));
          batch_tokens.clear();
        }
        pop_results(false /* blocking */);
      }

      if (!batch_tokens.empty())
        futures.emplace(post(batch_tokens, options));
      pop_results(true /* blocking */);
    }

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
    void work_loop(Translator& translator, size_t intra_threads);

    std::queue<std::pair<TranslationJob, std::promise<TranslationOutput>>> _work;
    std::vector<std::thread> _workers;
    std::vector<Translator> _translator_pool;
    std::mutex _mutex;
    std::condition_variable _cv;
    bool _request_end = false;
  };

}
