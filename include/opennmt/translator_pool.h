#pragma once

#include <future>
#include <mutex>
#include <thread>
#include <queue>

#include "translator.h"

namespace opennmt {

  using TranslationInput = std::vector<std::vector<std::string>>;
  using TranslationOutput = std::vector<std::vector<std::string>>;

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
