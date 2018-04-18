#pragma once

#include <ostream>

class Variable
{
public:
  Variable(unsigned short rank,
           const unsigned int* dimensions,
           const float* data)
    : _rank(rank)
    , _dimensions(dimensions)
    , _data(data) {
  }
  unsigned short rank() const {
    return _rank;
  }
  const unsigned int* dim() const {
    return _dimensions;
  }
  const float* data() const {
    return _data;
  }

  friend std::ostream& operator<<(std::ostream& os, const Variable& index);

private:
  unsigned short _rank;
  const unsigned int* _dimensions;
  const float* _data;
};

std::ostream& operator<<(std::ostream& os, const Variable& index) {
  os << '(';
  for (unsigned short i = 0; i < index._rank; ++i) {
    if (i > 0)
      os << ", ";
    os << index._dimensions[i];
  }
  os << ')';
  return os;
}
