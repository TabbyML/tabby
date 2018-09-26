#include "ctranslate2/translator_pool.h"

#include <fstream>

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

  std::future<TranslationOutput> TranslatorPool::post(const TranslationInput& batch_tokens,
                                                      const TranslationOptions& options) {
    std::future<TranslationOutput> future;

    {
      std::lock_guard<std::mutex> lock(_mutex);
      _work.emplace(std::piecewise_construct,
                    std::forward_as_tuple(),
                    std::forward_as_tuple(batch_tokens, options));
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
      auto& work = work_def.second;
      auto& input = work.first;
      auto& options = work.second;
      promise.set_value(translator.translate_batch(input, options));
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
      std::string token;
      for (size_t i = 0; i < line.length(); ++i) {
        if (line[i] == ' ') {
          if (!token.empty()) {
            tokens.emplace_back(std::move(token));
            token.clear();
          }
        } else {
          token += line[i];
        }
      }
      if (!token.empty())
        tokens.emplace_back(std::move(token));
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
