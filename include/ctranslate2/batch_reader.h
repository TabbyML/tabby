#pragma once

#include <istream>
#include <string>
#include <vector>

#include "translator.h"

namespace ctranslate2 {

  template <typename Reader>
  class BatchReader {
  public:
    BatchReader(std::istream& stream, Reader& reader)
      : _stream(stream)
      , _reader(reader)
      , _end(false)
    {
      advance();
    }

    bool has_next() const {
      return !_end;
    }

    std::vector<std::vector<std::string>>
    get_next(const size_t max_batch_size,
             const BatchType batch_type = BatchType::Examples) {
      std::vector<std::vector<std::string>> batch;
      batch.reserve(max_batch_size);

      size_t batch_size = 0;

      while (!_end) {
        const size_t batch_size_increment = get_batch_size_increment(peek_next_line(), batch_type);
        if (batch_size > 0 && batch_size + batch_size_increment > max_batch_size)
          break;
        batch.emplace_back(get_next_line());
        batch_size += batch_size_increment;
      }

      return batch;
    }

  private:
    std::istream& _stream;
    Reader& _reader;
    std::vector<std::string> _next;
    bool _end;

    void advance() {
      _next.clear();
      if (!_reader(_stream, _next)) {
        _end = true;
        _next.clear();
      }
    }

    const std::vector<std::string>& peek_next_line() {
      return _next;
    }

    std::vector<std::string> get_next_line() {
      auto next = std::move(_next);
      advance();
      return next;
    }
  };

}
