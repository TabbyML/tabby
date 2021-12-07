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

  template <typename T>
  static size_t get_batch_size_increment(const std::vector<T>& example,
                                         const BatchType batch_type) {
    switch (batch_type) {
    case BatchType::Tokens:
      return example.size();
    default:
      return 1;
    };
  }

  std::vector<std::vector<std::string>>
  BatchReader::get_next(const size_t max_batch_size,
                        const BatchType batch_type) {
    std::vector<std::vector<std::string>> batch;
    batch.reserve(max_batch_size);

    size_t batch_size = 0;

    while (has_next_element()) {
      const size_t batch_size_increment = get_batch_size_increment(peek_next_element(),
                                                                   batch_type);
      if (batch_size > 0 && batch_size + batch_size_increment > max_batch_size)
        break;
      batch.emplace_back(get_next_element());
      batch_size += batch_size_increment;
    }

    return batch;
  }

  VectorReader::VectorReader(std::vector<std::vector<std::string>> examples)
    : _examples(std::move(examples))
    , _index(0)
  {
  }

  bool VectorReader::has_next_element() const {
    return _index < _examples.size();
  }

  const std::vector<std::string>& VectorReader::peek_next_element() {
    return _examples[_index];
  }

  std::vector<std::string> VectorReader::get_next_element() {
    return std::move(_examples[_index++]);
  }

  void ParallelBatchReader::add(std::unique_ptr<BatchReader> reader) {
    _readers.emplace_back(std::move(reader));
  }

  std::vector<std::vector<std::vector<std::string>>>
  ParallelBatchReader::get_next(const size_t max_batch_size,
                                const BatchType batch_type) {
    std::vector<std::vector<std::vector<std::string>>> batches;
    batches.resize(_readers.size());
    batches[0] = _readers[0]->get_next(max_batch_size, batch_type);

    const size_t batch_size = batches[0].size();
    for (size_t i = 1; i < _readers.size(); ++i) {
      batches[i] = _readers[i]->get_next(batch_size);
      if (batches[i].size() != batch_size)
        throw std::runtime_error("One input stream has less elements than the others");
    }

    return batches;
  }


  std::vector<Batch>
  rebatch_input(const std::vector<std::vector<std::string>>& source,
                const std::vector<std::vector<std::string>>& target,
                size_t max_batch_size,
                BatchType batch_type,
                bool filter_empty) {
    if (!target.empty() && target.size() != source.size())
      throw std::invalid_argument("Batch size mismatch: got "
                                  + std::to_string(source.size()) + " for source and "
                                  + std::to_string(target.size()) + " for target");

    const size_t global_batch_size = source.size();
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
              [&source](size_t i1, size_t i2) {
                return source[i1].size() > source[i2].size();
              });

    // Ignore empty examples.
    // As example_index is sorted from longest to shortest, we simply pop empty examples
    // from the back.
    while (filter_empty && !example_index.empty() && source[example_index.back()].empty())
      example_index.pop_back();

    std::vector<Batch> batches;
    if (example_index.empty())
      return batches;
    batches.reserve(example_index.size());

    ParallelBatchReader batch_reader;
    batch_reader.add(std::make_unique<VectorReader>(index_vector(source, example_index)));
    if (!target.empty())
      batch_reader.add(std::make_unique<VectorReader>(index_vector(target, example_index)));

    for (size_t offset = 0;;) {
      auto batch_tokens = batch_reader.get_next(max_batch_size, batch_type);
      if (batch_tokens[0].empty())
        break;

      Batch batch;
      batch.source = std::move(batch_tokens[0]);
      if (batch_tokens.size() > 1)
        batch.target = std::move(batch_tokens[1]);

      const size_t batch_size = batch.source.size();
      batch.example_index.insert(batch.example_index.begin(),
                                 example_index.begin() + offset,
                                 example_index.begin() + offset + batch_size);
      offset += batch_size;

      batches.emplace_back(std::move(batch));
    }

    return batches;
  }

}
