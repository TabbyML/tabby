#pragma once

#include <iostream>

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <math.h>
#include <unistd.h>

#include <string>
#include <map>

#include "storage_view.h"

void* mmap_file(const char* path, size_t* file_size) {
  *file_size = 0;
  struct stat st;
  int s = stat(path, &st);
  if (s == -1)
    return nullptr;
  *file_size = st.st_size;
  int fd = open(path, O_RDONLY, 0);
  if (fd == -1)
    return nullptr;
  void* map = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
  close(fd);
  if (map == MAP_FAILED)
    return nullptr;
  return map;
}

template <typename T>
T consume(unsigned char** ptr) {
  T val = *reinterpret_cast<T*>(*ptr);
  *ptr += sizeof (T);
  return val;
}

class Model
{
public:
  Model(const std::string& path) {
    _model = mmap_file(path.c_str(), &_model_size);
    if (_model == nullptr)
      throw std::runtime_error("failed to load the model " + path);

    auto ptr = reinterpret_cast<unsigned char*>(_model);
    auto num_variables = consume<unsigned int>(&ptr);

    for (unsigned int i = 0; i < num_variables; ++i) {
      auto name_length = consume<unsigned short>(&ptr);
      auto name = reinterpret_cast<const char*>(ptr);
      ptr += name_length;
      unsigned short rank = consume<unsigned char>(&ptr);
      auto dimensions = reinterpret_cast<const unsigned int*>(ptr);
      unsigned int offset = 1;
      std::vector<size_t> shape(rank);
      for (unsigned int k = 0; k < rank; k++) {
        shape[k] = static_cast<size_t>(dimensions[k]);
        offset *= consume<unsigned int>(&ptr);
      }
      unsigned int data_width = consume<unsigned char>(&ptr);
      _variable_index.emplace(name, StorageView<float>(reinterpret_cast<float*>(ptr), shape));
      ptr += offset * data_width;
    }

    // for (const auto& index : _variable_index)
    //   std::cout << index.first << ": " << index.second << std::endl;
  }

  ~Model() {
    if (_model != nullptr)
      munmap(_model, _model_size);
  }

  const StorageView<float>& get_variable(const std::string& scope) const {
    auto it = _variable_index.lower_bound(scope);
    if (it->first.find(scope) == std::string::npos)
      throw std::out_of_range("no variable found in scope '" + scope + "'");
    return it->second;
  }

private:
  void* _model;
  size_t _model_size;
  std::map<std::string, StorageView<float> > _variable_index;
};
