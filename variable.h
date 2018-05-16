#pragma once

class Variable
{
public:
  Variable(const float* data, const std::vector<size_t>& shape)
    : _data(data)
    , _shape(shape) {
  }
  size_t rank() const {
    return _shape.size();
  }
  const std::vector<size_t>& shape() const {
    return _shape;
  }
  const float* data() const {
    return _data;
  }

private:
  const float* _data;
  std::vector<size_t> _shape;
};
