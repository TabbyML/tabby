#include "ctranslate2/batch_reader.h"

#include <algorithm>
#include <numeric>

#include "ctranslate2/utils.h"

namespace ctranslate2 {

  BatchType str_to_batch_type(const std::string& batch_type) {
    if (batch_type == "examples")
      return BatchType::Examples;
    else if (batch_type == "tokens")
      return BatchType::Tokens;
    throw std::invalid_argument("Invalid batch type: " + batch_type);
  }

  static inline size_t get_batch_size_increment(const Example& example,
                                                const BatchType batch_type) {
    switch (batch_type) {
    case BatchType::Tokens:
      return example.length();
    default:
      return 1;
    };
  }

  std::vector<std::vector<std::string>> Batch::get_stream(size_t index) const {
    std::vector<std::vector<std::string>> stream;
    if (examples.empty() || index >= examples.front().num_streams())
      return stream;
    stream.reserve(examples.size());
    for (const auto& example : examples)
      stream.emplace_back(example.streams[index]);
    return stream;
  }

  std::vector<Example>
  BatchReader::get_next(const size_t max_batch_size,
                        const BatchType batch_type) {
    if (max_batch_size == 0)
      throw std::invalid_argument("BatchReader: max_batch_size must be > 0");

    if (!_initialized) {
      _next = get_next_example();
      _initialized = true;
    }

    std::vector<Example> batch;
    if (_next.empty())
      return batch;

    batch.reserve(max_batch_size);

    size_t batch_size = 0;

    while (!_next.empty()) {
      const size_t batch_size_increment = get_batch_size_increment(_next, batch_type);
      if (batch_size > 0 && batch_size + batch_size_increment > max_batch_size)
        break;
      batch.emplace_back(std::move(_next));
      batch_size += batch_size_increment;
      _next = get_next_example();
    }

    return batch;
  }

  VectorReader::VectorReader(std::vector<std::vector<std::string>> examples)
  {
    _examples.reserve(examples.size());
    for (auto& example : examples)
      _examples.emplace_back(std::move(example));
  }

  VectorReader::VectorReader(std::vector<Example> examples)
    : _examples(std::move(examples))
  {
  }

  Example VectorReader::get_next_example() {
    if (_index >= _examples.size())
      return Example();
    return std::move(_examples[_index++]);
  }

  void ParallelBatchReader::add(std::unique_ptr<BatchReader> reader) {
    _readers.emplace_back(std::move(reader));
  }

  Example ParallelBatchReader::get_next_example() {
    Example example;

    for (const auto& reader : _readers) {
      auto stream_example = reader->get_next_example();

      if (example.empty()) {
        if (stream_example.empty())
          break;
        example.streams.reserve(_readers.size());
      } else if (stream_example.empty()) {
        throw std::runtime_error("One input stream has less examples than the others");
      }

      for (auto& stream : stream_example.streams)
        example.streams.emplace_back(std::move(stream));
    }

    return example;
  }

  size_t ParallelBatchReader::num_examples() const {
    for (const auto& reader : _readers) {
      const size_t num = reader->num_examples();
      if (num != 0)
        return num;
    }
    return 0;
  }


  std::vector<Example>
  load_examples(std::vector<std::vector<std::vector<std::string>>> streams) {
    ParallelBatchReader reader;

    for (auto& stream : streams) {
      if (stream.empty())
        continue;
      reader.add(std::make_unique<VectorReader>(std::move(stream)));
    }

    const size_t num_examples = reader.num_examples();
    if (num_examples == 0)
      return {};
    return reader.get_next(num_examples);
  }

  std::vector<Batch>
  rebatch_input(const std::vector<Example>& examples,
                size_t max_batch_size,
                BatchType batch_type) {
    if (examples.empty())
      return {};

    const size_t global_batch_size = examples.size();
    if (max_batch_size == 0) {
      max_batch_size = global_batch_size;
      batch_type = BatchType::Examples;
    }

    // Sorting the source inputs from the longest to the shortest has 2 benefits:
    //
    // 1. When max_batch_size is smaller that the number of inputs, we prefer translating
    //    together sentences that have a similar length for improved efficiency.
    // 2. Decoding functions remove finished translations from the batch. On CPU, arrays are
    //    updated in place so it is more efficient to remove content at the end. Shorter sentences
    //    are more likely to finish first so we sort the batch accordingly.
    std::vector<size_t> example_index(global_batch_size);
    std::iota(example_index.begin(), example_index.end(), 0);
    std::sort(example_index.begin(), example_index.end(),
              [&examples](size_t i1, size_t i2) {
                return examples[i1].length() > examples[i2].length();
              });

    std::vector<Batch> batches;
    if (example_index.empty())
      return batches;
    batches.reserve(example_index.size());

    VectorReader batch_reader(index_vector(examples, example_index));

    for (size_t offset = 0;;) {
      auto examples_part = batch_reader.get_next(max_batch_size, batch_type);
      if (examples_part.empty())
        break;

      const size_t batch_size = examples_part.size();

      Batch batch;
      batch.examples = std::move(examples_part);
      batch.example_index.insert(batch.example_index.begin(),
                                 example_index.begin() + offset,
                                 example_index.begin() + offset + batch_size);
      offset += batch_size;

      batches.emplace_back(std::move(batch));
    }

    return batches;
  }

}
