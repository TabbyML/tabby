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

  std::future<std::vector<TranslationResult>>
  TranslatorPool::translate_batch_async(std::vector<std::vector<std::string>> source,
                                        TranslationOptions options) {
    return post(std::move(source), std::move(options));
  }

  std::future<std::vector<TranslationResult>>
  TranslatorPool::translate_batch_async(std::vector<std::vector<std::string>> source,
                                        std::vector<std::vector<std::string>> target_prefix,
                                        TranslationOptions options) {
    return post(std::move(source), std::move(target_prefix), std::move(options));
  }

  std::future<std::vector<TranslationResult>>
  TranslatorPool::post(std::vector<std::vector<std::string>> source,
                       TranslationOptions options,
                       bool throttle) {
    return post(std::move(source),
                std::vector<std::vector<std::string>>(),
                std::move(options),
                throttle);
  }

  std::future<std::vector<TranslationResult>>
  TranslatorPool::post(std::vector<std::vector<std::string>> source,
                       std::vector<std::vector<std::string>> target_prefix,
                       TranslationOptions options,
                       bool throttle) {
    auto* job = new TranslationJob(std::move(source),
                                   std::move(target_prefix),
                                   std::move(options));
    auto future = job->get_future();
    post_job(std::unique_ptr<Job>(job), throttle);
    return future;
  }

  void TranslatorPool::post_job(std::unique_ptr<Job> job, bool throttle) {
    std::unique_lock<std::mutex> lock(_mutex);
    if (throttle)
      _can_add_more_work.wait(lock, [this]{ return _work.size() < 2 * _workers.size(); });

    // locked again here

    _work.emplace(std::move(job));

    lock.unlock();
    _cv.notify_one();
  }

  std::vector<TranslationResult>
  TranslatorPool::translate_batch(const std::vector<std::vector<std::string>>& source,
                                  const TranslationOptions& options) {
    return translate_batch(source, std::vector<std::vector<std::string>>(), options);
  }

  std::vector<TranslationResult>
  TranslatorPool::translate_batch(const std::vector<std::vector<std::string>>& source,
                                  const std::vector<std::vector<std::string>>& target_prefix,
                                  const TranslationOptions& user_options) {
    TranslationOptions options = user_options;
    options.validate();
    options.validated = true;

    if (source.empty())
      return std::vector<TranslationResult>();

    // Rebatch the input and post each sub-batch in the translation queue.
    auto batches = rebatch_input(source, target_prefix, options);
    options.rebatch_input = false;

    std::vector<std::future<std::vector<TranslationResult>>> futures;
    futures.reserve(batches.size());
    for (auto& batch : batches) {
      futures.emplace_back(post(std::move(batch.source),
                                std::move(batch.target),
                                options));
    }

    const TranslationResult empty_result(options.num_hypotheses, options.return_attention);
    std::vector<TranslationResult> results(source.size(), empty_result);

    // Wait for the result of each sub-batch.
    for (size_t batch_id = 0; batch_id < batches.size(); ++batch_id) {
      auto batch_results = futures[batch_id].get();
      for (size_t i = 0; i < batch_results.size(); ++i)
        results[batches[batch_id].example_index[i]] = std::move(batch_results[i]);
    }

    return results;
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
        // The CUDA context is destroyed when the thread exits, so we clear the translation
        // resources now when the CUDA context is still active.
        translator.detach_model();
        break;
      }

      auto job = std::move(_work.front());
      _work.pop();
      lock.unlock();

      _can_add_more_work.notify_one();

      job->run(translator);
    }
  }

  template <typename OutputType>
  void TranslatorPool::BaseJob<OutputType>::run(Translator& translator) {
    try {
      _promise.set_value(compute(translator));
    } catch (...) {
      try {
        // Store the exception in the shared state so that future.get() will throw it.
        _promise.set_exception(std::current_exception());
      } catch (...) {
        // set_exception may throw too.
      }
    }
  }

  std::vector<TranslationResult>
  TranslatorPool::TranslationJob::compute(Translator& translator) const {
    return translator.translate_batch_with_prefix(_source, _target_prefix, _options);
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

  TranslationStats TranslatorPool::consume_text_file(const std::string& source_file,
                                                     const std::string& output_file,
                                                     size_t read_batch_size,
                                                     const TranslationOptions& options,
                                                     bool with_scores,
                                                     const std::string* target_file) {
    std::ifstream source;
    open_input_file(source_file, source);
    std::ofstream output;
    open_output_file(output_file, output);

    std::unique_ptr<std::ifstream> target;
    if (target_file) {
      target.reset(new std::ifstream());
      open_input_file(*target_file, *target);
    }

    return consume_text_file(source, output, read_batch_size, options, with_scores, target.get());
  }

  TranslationStats TranslatorPool::consume_text_file(std::istream& source,
                                                     std::ostream& output,
                                                     size_t read_batch_size,
                                                     const TranslationOptions& options,
                                                     bool with_scores,
                                                     std::istream* target) {
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

    return consume_raw_text_file(source,
                                 target,
                                 output,
                                 tokenizer,
                                 tokenizer,
                                 detokenizer,
                                 read_batch_size,
                                 options,
                                 with_scores);
  }

  size_t TranslatorPool::num_queued_batches() {
    const std::lock_guard<std::mutex> lock(_mutex);
    return _work.size();
  }

  size_t TranslatorPool::num_translators() const {
    return _translators.size();
  }

  const std::vector<Translator>& TranslatorPool::get_translators() const {
    return _translators;
  }

}
