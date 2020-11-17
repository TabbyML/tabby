#pragma once

#include <istream>
#include <memory>
#include <string>
#include <vector>

namespace ctranslate2 {

  enum class BatchType {
    Examples,
    Tokens,
  };

  BatchType str_to_batch_type(const std::string& batch_type);

  template <typename T>
  size_t get_batch_size_increment(const std::vector<T>& example,
                                  const BatchType batch_type) {
    switch (batch_type) {
    case BatchType::Tokens:
      return example.size();
    default:
      return 1;
    };
  }

  template <typename T>
  size_t get_batch_size(const std::vector<std::vector<T>>& examples,
                        const BatchType batch_type) {
    size_t batch_size = 0;
    for (const std::vector<T>& example : examples)
      batch_size += get_batch_size_increment(example, batch_type);
    return batch_size;
  }

  // Base class to produce batches.
  class BatchReader {
  public:
    virtual ~BatchReader() = default;

    std::vector<std::vector<std::string>>
    get_next(const size_t max_batch_size,
             const BatchType batch_type = BatchType::Examples);

    // Returns true if there are still elements to read.
    virtual bool has_next() const = 0;

  protected:
    // Returns the next element but does not consume it.
    virtual const std::vector<std::string>& peek_next_element() = 0;

    // Consumes and returns the next element.
    virtual std::vector<std::string> get_next_element() = 0;
  };

  // Read batches from a stream.
  template <typename Reader>
  class StreamReader : public BatchReader {
  public:
    StreamReader(std::istream& stream, Reader& reader)
      : _stream(stream)
      , _reader(reader)
      , _end(false)
    {
      advance();
    }

    bool has_next() const override {
      return !_end;
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

  protected:
    const std::vector<std::string>& peek_next_element() override {
      return _next;
    }

    std::vector<std::string> get_next_element() override {
      auto next = std::move(_next);
      advance();
      return next;
    }
  };

  // Read batches from a vector of elements.
  class VectorReader : public BatchReader {
  public:
    VectorReader(std::vector<std::vector<std::string>> examples);
    bool has_next() const override;

  protected:
    const std::vector<std::string>& peek_next_element() override;
    std::vector<std::string> get_next_element() override;

  private:
    std::vector<std::vector<std::string>> _examples;
    size_t _index;
  };

  // Read batches from multiple sources.
  class ParallelBatchReader {
  public:
    void add(BatchReader* reader);  // The instance takes ownership of the pointer.

    // batch_type is applied to the first stream and then the function takes the same
    // number of examples from the other streams.
    std::vector<std::vector<std::vector<std::string>>>
    get_next(const size_t max_batch_size,
             const BatchType batch_type = BatchType::Examples);

    bool has_next() const;

  private:
    std::vector<std::unique_ptr<BatchReader>> _readers;
  };

}
