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

  class TranslatorPool {
  public:
    template <typename... Args>
    TranslatorPool(size_t num_replicas, Args&&... args) {
      _translator_pool.emplace_back(std::forward<Args>(args)...);
      for (size_t i = 1; i < num_replicas; ++i)
        _translator_pool.emplace_back(_translator_pool.front());
      for (auto& translator : _translator_pool)
        _workers.emplace_back(&TranslatorPool::work_loop, this, std::ref(translator));
    }
    ~TranslatorPool();

    std::future<TranslationOutput> post(const TranslationInput& batch_tokens);

    template <typename Preprocessor, typename Postprocessor>
    void consume_text_stream(std::istream& in,
                             std::ostream& out,
                             size_t max_batch_size,
                             Preprocessor& preprocess,
                             Postprocessor& postprocess) {
      std::queue<std::future<TranslationOutput>> futures;

      auto pop_results = [&futures, &out, &postprocess](bool blocking) {
        static const auto zero_sec = std::chrono::seconds(0);
        while (!futures.empty()
               && (blocking
                   || futures.front().wait_for(zero_sec) == std::future_status::ready)) {
          for (const auto& result : futures.front().get())
            out << postprocess(result.output()) << std::endl;
          futures.pop();
        }
      };

      TranslationInput batch_tokens;
      std::string line;

      while (std::getline(in, line)) {
        batch_tokens.emplace_back(preprocess(line));
        if (batch_tokens.size() == max_batch_size) {
          futures.emplace(post(batch_tokens));
          batch_tokens.clear();
        }
        pop_results(false /* blocking */);
      }

      if (!batch_tokens.empty())
        futures.emplace(post(batch_tokens));
      pop_results(true /* blocking */);
    }

  private:
    void work_loop(Translator& translator);

    std::queue<std::pair<std::promise<TranslationOutput>, TranslationInput>> _work;
    std::vector<std::thread> _workers;
    std::vector<Translator> _translator_pool;
    std::mutex _mutex;
    std::condition_variable _cv;
    bool _request_end = false;
  };

}
