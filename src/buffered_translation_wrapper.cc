#include "ctranslate2/buffered_translation_wrapper.h"

namespace ctranslate2 {

  BufferedTranslationWrapper::BufferedTranslationWrapper(std::shared_ptr<Translator> translator,
                                                         size_t max_batch_size,
                                                         size_t buffer_timeout_in_micros,
                                                         TranslationOptions options,
                                                         size_t max_buffer_size)
    : _translator(std::move(translator))
    , _options(std::move(options))
    , _max_batch_size(max_batch_size)
    , _max_buffer_size(max_buffer_size == 0 ? max_batch_size : max_buffer_size)
    , _buffer_timeout(buffer_timeout_in_micros)
  {
    _background_thread = std::make_unique<std::thread>(&BufferedTranslationWrapper::buffer_loop,
                                                       this);
  }

  BufferedTranslationWrapper::~BufferedTranslationWrapper() {
    {
      const std::lock_guard<std::mutex> lock(_mutex);
      _stop = true;
    }
    _cv.notify_one();
    _background_thread->join();
  }

  std::future<TranslationResult>
  BufferedTranslationWrapper::translate_async(std::vector<std::string> source,
                                              std::vector<std::string> target) {
    std::promise<TranslationResult> promise;
    auto future = promise.get_future();

    bool notify = false;

    {
      const std::lock_guard<std::mutex> lock(_mutex);

      _promises.emplace(std::move(promise));
      _examples.emplace(std::move(source), std::move(target));

      notify = (_examples.size() >= _max_buffer_size);
    }

    if (notify)
      _cv.notify_one();

    return future;
  }

  std::vector<std::future<TranslationResult>>
  BufferedTranslationWrapper::translate_batch_async(std::vector<std::vector<std::string>> source,
                                                    std::vector<std::vector<std::string>> target) {
    std::vector<std::future<TranslationResult>> futures;
    futures.reserve(source.size());

    for (size_t i = 0; i < source.size(); ++i) {
      futures.emplace_back(translate_async(std::move(source[i]),
                                           target.empty()
                                           ? std::vector<std::string>()
                                           : std::move(target[i])));
    }

    return futures;
  }

  void BufferedTranslationWrapper::buffer_loop() {
    while (true) {
      std::unique_lock<std::mutex> lock(_mutex);
      _cv.wait_for(lock, _buffer_timeout,
                   [this]{ return _examples.size() >= _max_buffer_size || _stop; });

      // Get the stop flag value when we hold the lock.
      const bool stop = _stop;

      if (!_examples.empty()) {
        // Build full batches unless the timeout is reached or we are stopping the process.
        size_t flush_size = _examples.size();
        if (!stop && flush_size > _max_batch_size)
          flush_size -= flush_size % _max_batch_size;

        std::vector<Example> examples;
        std::vector<std::promise<TranslationResult>> promises;
        examples.reserve(flush_size);
        promises.reserve(flush_size);

        for (size_t i = 0; i < flush_size; ++i) {
          examples.emplace_back(std::move(_examples.front()));
          promises.emplace_back(std::move(_promises.front()));
          _examples.pop();
          _promises.pop();
        }

        // Release the lock as soon as the buffer is flushed.
        lock.unlock();

        _translator->post_examples(
          examples,
          _max_batch_size,
          BatchType::Examples,
          std::move(promises),
          [this](models::SequenceToSequenceReplica& model, const Batch& batch) {
            return run_translation(model, batch, _options);
          });
      }

      if (stop)
        break;
    }
  }

}
