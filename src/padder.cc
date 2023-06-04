#include "ctranslate2/padder.h"

#include <algorithm>

namespace ctranslate2 {

  Padder::Padder(const StorageView& lengths,
                 const dim_t max_time,
                 const dim_t pad_batch_to_multiple)
    : _batch_size(lengths.size()) {
    const std::vector<int32_t> lengths_vec = lengths.to_vector<int32_t>();
    if (max_time < 0)
      _max_time = *std::max_element(lengths_vec.begin(), lengths_vec.end());
    else
      _max_time = max_time;
    const bool has_padding = std::any_of(lengths_vec.begin(),
                                         lengths_vec.end(),
                                         [this](const int32_t length) {
                                           return length != _max_time;
                                         });
    if (!has_padding)
      return;

    const dim_t max_size = _max_time * _batch_size;
    std::vector<int32_t> padded_to_flat;
    std::vector<int32_t> flat_to_padded;
    padded_to_flat.reserve(max_size);
    flat_to_padded.reserve(max_size);

    dim_t padded_offset = 0;
    dim_t flat_offset = 0;

    for (dim_t i = 0; i < _batch_size; ++i) {
      const dim_t length = lengths_vec[i];
      for (dim_t t = 0; t < length; ++t) {
        padded_to_flat.push_back(padded_offset + t);
        flat_to_padded.push_back(flat_offset + t);
      }
      for (dim_t t = length; t < _max_time; ++t) {
        flat_to_padded.push_back(flat_offset + length - 1);
      }
      padded_offset += _max_time;
      flat_offset += length;
    }

    while (padded_to_flat.size() % pad_batch_to_multiple != 0) {
      padded_to_flat.push_back(padded_to_flat.back());
      ++flat_offset;
    }

    const Device device = lengths.device();
    _padded_to_flat = StorageView({flat_offset}, padded_to_flat, device);
    _flat_to_padded = StorageView({padded_offset}, flat_to_padded, device);
  }

  void Padder::remove_padding(StorageView& x) const {
    if (!_padded_to_flat)
      return;
    Shape shape = x.shape();
    shape[1] *= shape[0];
    shape.erase(shape.begin());
    x.reshape(std::move(shape));
    _gather_op(x, _padded_to_flat);
  }

  void Padder::add_padding(StorageView& x) const {
    if (!_flat_to_padded)
      return;
    _gather_op(x, _flat_to_padded);
    Shape shape = x.shape();
    shape[0] /= _batch_size;
    shape.insert(shape.begin(), _batch_size);
    x.reshape(std::move(shape));
  }


}
