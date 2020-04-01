#include "ctranslate2/translator_pool.h"

#include <chrono>
#include <fstream>

#include "ctranslate2/utils.h"

namespace ctranslate2 {

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

  void TranslatorPool::work_loop(Translator& translator, size_t intra_threads) {
    // set_num_threads is called here because it sets the number of OpenMP threads for
    // the current thread.
    set_num_threads(intra_threads);

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

  TranslationStats TranslatorPool::consume_text_file(const std::string& in_file,
                                                     const std::string& out_file,
                                                     size_t read_batch_size,
                                                     const TranslationOptions& options,
                                                     bool with_scores) {
    std::ifstream in(in_file);
    if (!in.is_open())
      throw std::runtime_error("failed to open input file " + in_file);
    std::ofstream out(out_file);
    if (!out.is_open())
      throw std::runtime_error("failed to open output file " + out_file);
    return consume_text_file(in, out, read_batch_size, options, with_scores);
  }

  TranslationStats TranslatorPool::consume_text_file(std::istream& in,
                                                     std::ostream& out,
                                                     size_t read_batch_size,
                                                     const TranslationOptions& options,
                                                     bool with_scores) {
    TranslationStats stats;

    auto reader = [](std::istream& in, std::vector<std::string>& tokens) {
      std::string line;
      if (!std::getline(in, line))
        return false;
      tokens = split_string(line, ' ');
      return true;
    };

    auto writer = [&stats, &with_scores](std::ostream& out, const TranslationResult& result) {
      const auto& hypotheses = result.hypotheses();
      const auto& scores = result.scores();
      stats.num_examples += 1;
      stats.num_tokens += hypotheses[0].size();
      for (size_t n = 0; n < hypotheses.size(); ++n) {
        if (with_scores)
          out << (result.has_scores() ? scores[n] : 0) << " ||| ";
        for (size_t i = 0; i < hypotheses[n].size(); ++i) {
          if (i > 0)
            out << ' ';
          out << hypotheses[n][i];
        }
        out << '\n';
      }
    };

    const auto t1 = std::chrono::high_resolution_clock::now();

    consume_stream(in, out, read_batch_size, options, reader, writer);
    out.flush();

    const auto t2 = std::chrono::high_resolution_clock::now();
    stats.total_time_in_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
      t2 - t1).count();
    return stats;
  }

  const std::vector<Translator>& TranslatorPool::get_translators() const {
    return _translator_pool;
  }

}
