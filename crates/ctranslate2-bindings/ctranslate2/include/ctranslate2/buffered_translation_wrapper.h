#pragma once

#include "translator.h"

namespace ctranslate2 {

  // This class wraps a Translator instance and bufferizes incoming translation requests.
  // The buffer is flushed when one of the following conditions is met:
  //
  //  * buffer_timeout_in_micros microseconds have passed
  //  * max_buffer_size examples are ready to be translated
  //
  // By default, max_buffer_size is set to max_batch_size, but it can be set to a larger value
  // in which case the buffer content is sorted by length and rebatched according to max_batch_size.
  class BufferedTranslationWrapper {
  public:
    BufferedTranslationWrapper(std::shared_ptr<Translator> translator,
                               size_t max_batch_size,
                               size_t buffer_timeout_in_micros,
                               TranslationOptions options = TranslationOptions(),
                               size_t max_buffer_size = 0);
    ~BufferedTranslationWrapper();

    std::future<TranslationResult>
    translate_async(std::vector<std::string> source, std::vector<std::string> target = {});

    std::vector<std::future<TranslationResult>>
    translate_batch_async(std::vector<std::vector<std::string>> source,
                          std::vector<std::vector<std::string>> target = {});

  private:
    std::shared_ptr<Translator> _translator;
    const TranslationOptions _options;
    const size_t _max_batch_size;
    const size_t _max_buffer_size;
    const std::chrono::microseconds _buffer_timeout;
    std::unique_ptr<std::thread> _background_thread;
    bool _stop = false;

    std::mutex _mutex;
    std::condition_variable _cv;
    std::queue<Example> _examples;
    std::queue<std::promise<TranslationResult>> _promises;

    void buffer_loop();
  };

}
