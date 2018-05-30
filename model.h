#pragma once

#include <fstream>
#include <string>
#include <map>

#include "storage_view.h"

namespace onmt {

  template <typename T>
  T consume(std::ifstream& in) {
    T val;
    in.read(reinterpret_cast<char*>(&val), sizeof (T));
    return val;
  }

  template <typename T>
  T* consume(std::ifstream& in, size_t n) {
    T* data = new T[n];
    in.read(reinterpret_cast<char*>(data), n * sizeof (T));
    return data;
  }

  class Model
  {
  public:
    Model(const std::string& path) {
      std::ifstream model(path, std::ios_base::in | std::ios_base::binary);
      if (!model.is_open())
        throw std::runtime_error("failed to load the model " + path);

      auto num_variables = consume<uint32_t>(model);

      for (uint32_t i = 0; i < num_variables; ++i) {
        auto name_length = consume<uint16_t>(model);
        auto name = consume<char>(model, name_length);
        auto rank = consume<uint8_t>(model);
        auto dimensions = consume<uint32_t>(model, rank);
        auto data_width = consume<uint8_t>(model);
        auto data_size = consume<uint32_t>(model);
        auto data = consume<char>(model, data_size * data_width);

        std::vector<size_t> shape(rank);
        for (unsigned int k = 0; k < rank; k++) {
          shape[k] = static_cast<size_t>(dimensions[k]);
        }

        StorageView* view;

        if (data_width == 4) {
          view = new StorageView(reinterpret_cast<float*>(data), shape);
        } else if (data_width == 2) {
          view = new StorageView(reinterpret_cast<int16_t*>(data), shape);
        }

        _variable_index.emplace(std::piecewise_construct,
                                std::forward_as_tuple(name),
                                std::forward_as_tuple(*view));

        delete view;
        delete [] name;
        delete [] dimensions;
        delete [] data;
      }
    }

    const StorageView& get_variable(const std::string& scope) const {
      auto it = _variable_index.lower_bound(scope);
      if (it->first.find(scope) == std::string::npos)
        throw std::out_of_range("no variable found in scope '" + scope + "'");
      return it->second;
    }

  private:
    std::map<std::string, StorageView> _variable_index;
  };

}
