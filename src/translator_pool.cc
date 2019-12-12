#include "ctranslate2/translator_pool.h"

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
                                                      const TranslationOptions& options) {
    TranslationInput target_prefix;
    return post(source, target_prefix, options);
  }

  std::future<TranslationOutput> TranslatorPool::post(const TranslationInput& source,
                                                      const TranslationInput& target_prefix,
                                                      const TranslationOptions& options) {
    std::future<TranslationOutput> future;
    TranslationJob job;
    job.source = source;
    job.target_prefix = target_prefix;
    job.options = options;

    {
      std::lock_guard<std::mutex> lock(_mutex);
      _work.emplace(std::piecewise_construct,
                    std::forward_as_tuple(std::move(job)),
                    std::forward_as_tuple());
      future = _work.back().second.get_future();
    }

    _cv.notify_one();
    return future;
  }

  void TranslatorPool::work_loop(Translator& translator, size_t intra_threads) {
    auto& work_queue = _work;
    auto& end_requested = _request_end;

    // set_num_threads is called here because it sets the number of OpenMP threads for
    // the current thread.
    set_num_threads(intra_threads);

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

      auto& job = work_def.first;
      auto& promise = work_def.second;
      promise.set_value(translator.translate_batch_with_prefix(job.source,
                                                               job.target_prefix,
                                                               job.options));
    }
  }

  size_t TranslatorPool::consume_text_file(const std::string& in_file,
                                           const std::string& out_file,
                                           size_t max_batch_size,
                                           const TranslationOptions& options,
                                           bool with_scores) {
    std::ifstream in(in_file);
    if (!in.is_open())
      throw std::runtime_error("failed to open input file " + in_file);
    std::ofstream out(out_file);
    if (!out.is_open())
      throw std::runtime_error("failed to open output file " + out_file);
    return consume_text_file(in, out, max_batch_size, options, with_scores);
  }

  size_t TranslatorPool::consume_text_file(std::istream& in,
                                           std::ostream& out,
                                           size_t max_batch_size,
                                           const TranslationOptions& options,
                                           bool with_scores) {
    size_t num_tokens = 0;

    auto reader = [](std::istream& in, std::vector<std::string>& tokens) {
      std::string line;
      if (!std::getline(in, line))
        return false;
      tokens = split_string(line, ' ');
      return true;
    };

    auto writer = [&num_tokens, &with_scores](std::ostream& out, const TranslationResult& result) {
      const auto& hypotheses = result.hypotheses();
      const auto& scores = result.scores();
      num_tokens += hypotheses[0].size();
      for (size_t n = 0; n < hypotheses.size(); ++n) {
        if (with_scores)
          out << scores[n] << " ||| ";
        for (size_t i = 0; i < hypotheses[n].size(); ++i) {
          if (i > 0)
            out << ' ';
          out << hypotheses[n][i];
        }
        out << std::endl;
      }
    };

    consume_stream(in, out, max_batch_size, options, reader, writer);
    return num_tokens;
  }

}
