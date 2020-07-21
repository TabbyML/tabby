#include "ctranslate2/translator_pool.h"

#include "ctranslate2/utils.h"

namespace ctranslate2 {

  TranslatorPool::TranslatorPool(size_t num_translators,
                                 size_t num_threads_per_translator,
                                 const std::shared_ptr<const models::Model>& model) {
    create_translators(model, num_translators, num_threads_per_translator);
  }

  TranslatorPool::~TranslatorPool() {
    {
      std::lock_guard<std::mutex> lock(_mutex);
      _request_end = true;
    }
    _cv.notify_all();  // Request all workers to end their loop.
    for (auto& worker : _workers)
      worker.join();
  }

  std::future<TranslationOutput> TranslatorPool::post(const TranslationInput& source,
                                                      const TranslationOptions& options,
                                                      bool blocking) {
    TranslationInput target_prefix;
    return post(source, target_prefix, options, blocking);
  }

  std::future<TranslationOutput> TranslatorPool::post(const TranslationInput& source,
                                                      const TranslationInput& target_prefix,
                                                      const TranslationOptions& options,
                                                      bool blocking) {
    std::unique_lock<std::mutex> lock(_mutex);
    if (blocking)
      _can_add_more_work.wait(lock, [this]{ return _work.size() < 2 * _workers.size(); });

    // locked again here

    _work.emplace(std::piecewise_construct,
                  std::forward_as_tuple(source, target_prefix, options),
                  std::forward_as_tuple());

    std::future<TranslationOutput> future = _work.back().second.get_future();

    lock.unlock();
    _cv.notify_one();
    return future;
  }

  void TranslatorPool::create_translators(const std::shared_ptr<const models::Model>& model,
                                          size_t num_translators,
                                          size_t num_threads_per_translator) {
    if (model->device() == Device::CUDA) {
      // On GPU, we currently don't benefit much from running translators in parallel, even
      // when using separate streams. This could be revisited/improved in the future.
      num_translators = 1;
      // Most computation will run on GPU so multiple CPU computation threads are not useful.
      num_threads_per_translator = 1;
    }

    static const int core_offset = read_int_from_env("CT2_TRANSLATORS_CORE_OFFSET", -1);
    if (core_offset >= 0) {
#ifdef __linux__
      if (num_threads_per_translator > 1) {
        throw std::invalid_argument("Pinning translators to CPU cores requires intra_threads = 1");
      }
#else
      throw std::invalid_argument("Pinning translators to CPU cores is only supported on Linux");
#endif
    }

    _translators.reserve(num_translators);
    _workers.reserve(num_translators);
    for (size_t i = 0; i < num_translators; ++i) {
      _translators.emplace_back(model);
      _workers.emplace_back(&TranslatorPool::work_loop,
                            this,
                            std::ref(_translators.back()),
                            num_threads_per_translator);
#ifdef __linux__
      if (core_offset >= 0) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_offset + i, &cpuset);
        const int status = pthread_setaffinity_np(_workers.back().native_handle(),
                                                  sizeof (cpu_set_t),
                                                  &cpuset);
        if (status != 0) {
          throw std::runtime_error("Error calling pthread_setaffinity_np: "
                                   + std::to_string(status));
        }
      }
#endif
    }
  }

  void TranslatorPool::work_loop(Translator& translator, size_t num_threads) {
    // set_num_threads is called here because it sets the number of OpenMP threads for
    // the current thread.
    set_num_threads(num_threads);

    while (true) {
      std::unique_lock<std::mutex> lock(_mutex);
      _cv.wait(lock, [this]{ return !_work.empty() || _request_end; });

      if (_request_end) {
        lock.unlock();
        break;
      }

      auto work_def = std::move(_work.front());
      _work.pop();
      lock.unlock();

      _can_add_more_work.notify_one();

      const auto& job = work_def.first;
      auto& promise = work_def.second;
      try {
        promise.set_value(translator.translate_batch_with_prefix(job.source,
                                                                 job.target_prefix,
                                                                 job.options));
      } catch (...) {
        try {
          // Store the exception in the shared state so that future.get() will throw it.
          promise.set_exception(std::current_exception());
        } catch (...) {
          // set_exception may throw too.
        }
      }
    }
  }

  void TranslatorPool::open_input_file(const std::string& file, std::ifstream& stream) const {
    stream.open(file);
    if (!stream)
      throw std::runtime_error("failed to open input file " + file);
  }

  void TranslatorPool::open_output_file(const std::string& file, std::ofstream& stream) const {
    stream.open(file);
    if (!stream)
      throw std::runtime_error("failed to open output file " + file);
  }

  TranslationStats TranslatorPool::consume_text_file(const std::string& in_file,
                                                     const std::string& out_file,
                                                     size_t read_batch_size,
                                                     const TranslationOptions& options,
                                                     bool with_scores) {
    std::ifstream in;
    open_input_file(in_file, in);
    std::ofstream out;
    open_output_file(out_file, out);
    return consume_text_file(in, out, read_batch_size, options, with_scores);
  }

  TranslationStats TranslatorPool::consume_text_file(std::istream& in,
                                                     std::ostream& out,
                                                     size_t read_batch_size,
                                                     const TranslationOptions& options,
                                                     bool with_scores) {
    const auto tokenizer = [](const std::string& text) {
      return split_string(text, ' ');
    };

    const auto detokenizer = [](const std::vector<std::string>& tokens) {
      std::string text;
      for (const auto& token : tokens) {
        if (!text.empty())
          text += ' ';
        text += token;
      }
      return text;
    };

    return consume_raw_text_file(in,
                                 out,
                                 tokenizer,
                                 detokenizer,
                                 read_batch_size,
                                 options,
                                 with_scores);
  }

  const std::vector<Translator>& TranslatorPool::get_translators() const {
    return _translators;
  }

}
