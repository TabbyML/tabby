#include "ctranslate2/batch_reader.h"

namespace ctranslate2 {

  BatchType str_to_batch_type(const std::string& batch_type) {
    if (batch_type == "examples")
      return BatchType::Examples;
    else if (batch_type == "tokens")
      return BatchType::Tokens;
    throw std::invalid_argument("Invalid batch type: " + batch_type);
  }

  std::vector<std::vector<std::string>>
  BatchReader::get_next(const size_t max_batch_size,
                        const BatchType batch_type) {
    std::vector<std::vector<std::string>> batch;
    batch.reserve(max_batch_size);

    size_t batch_size = 0;

    while (has_next()) {
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

  bool VectorReader::has_next() const {
    return _index < _examples.size();
  }

  const std::vector<std::string>& VectorReader::peek_next_element() {
    return _examples[_index];
  }

  std::vector<std::string> VectorReader::get_next_element() {
    return std::move(_examples[_index++]);
  }

  void ParallelBatchReader::add(BatchReader* reader) {
    _readers.emplace_back(reader);
  }

  std::vector<std::vector<std::vector<std::string>>>
  ParallelBatchReader::get_next(const size_t max_batch_size,
                                const BatchType batch_type) {
    std::vector<std::vector<std::vector<std::string>>> batches;
    batches.resize(_readers.size());

    if (has_next()) {
      batches[0] = _readers[0]->get_next(max_batch_size, batch_type);

      const size_t batch_size = batches[0].size();
      for (size_t i = 1; i < _readers.size(); ++i) {
        batches[i] = _readers[i]->get_next(batch_size);
        if (batches[i].size() != batch_size)
          throw std::runtime_error("One input stream has less elements than the others");
      }
    }

    return batches;
  }

  bool ParallelBatchReader::has_next() const {
    for (const auto& reader : _readers) {
      if (reader->has_next())
        return true;
    }
    return false;
  }

}
