#include "ctranslate2/translator_pool.h"

namespace ctranslate2 {

  TranslatorPool::~TranslatorPool() {
    {
      std::lock_guard<std::mutex> lock(_mutex);
      _request_end = true;
    }
    _cv.notify_all();
    for (auto& worker : _workers)
      worker.join();
  }

  std::future<TranslationOutput> TranslatorPool::post(const TranslationInput& batch_tokens) {
    std::future<TranslationOutput> future;

    {
      std::lock_guard<std::mutex> lock(_mutex);
      _work.emplace(std::promise<TranslationOutput>(), batch_tokens);
      future = std::move(_work.back().first.get_future());
    }

    _cv.notify_one();
    return future;
  }

  void TranslatorPool::work_loop(Translator& translator) {
    auto& work_queue = _work;
    auto& end_requested = _request_end;

    while (true) {
      std::unique_lock<std::mutex> lock(_mutex);
      _cv.wait(lock, [&work_queue, &end_requested]{
          return !work_queue.empty() || end_requested;
      });

      if (end_requested) {
        lock.unlock();
        break;
      }

      auto work_def = std::move(work_queue.front());
      work_queue.pop();
      lock.unlock();

      auto& promise = work_def.first;
      auto& input = work_def.second;
      promise.set_value(translator.translate_batch(input));
    }
  }

}
