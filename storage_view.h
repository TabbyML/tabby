#pragma once

#include <iostream>

#include <algorithm>
#include <cassert>
#include <ostream>
#include <stdexcept>
#include <typeinfo>
#include <vector>

using Shape = std::vector<size_t>;

template <typename T>
class StorageView
{
public:
  StorageView() {
  }

  StorageView(const Shape& shape, T init = T()) {
    resize(shape);
    fill(init);
  }

  StorageView(T* data, const Shape& shape) {
    assign(data, shape);
  }

  StorageView(const StorageView& other) {
    assign(other);
  }

  StorageView(StorageView&& other) {
    assign(std::move(other));
  }

  ~StorageView() {
    release();
  }

  T* data() {
    return _data;
  }
  const T* data() const {
    return _data;
  }

  T* index(const std::vector<size_t>& indices) {
    return const_cast<T*>(static_cast<const StorageView&>(*this).index(indices));
  }
  const T* index(const std::vector<size_t>& indices) const {
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i)
      offset += indices[i] * _strides[i];
    assert(offset < _size);
    return _data + offset;
  }

  T& operator[](size_t index) {
    return const_cast<T&>(static_cast<const StorageView&>(*this)[index]);
  }
  const T& operator[](size_t index) const {
    assert(index < _size);
    return _data[index];
  }
  T& operator[](const std::vector<size_t>& indices) {
    return index(indices)[0];
  }
  const T& operator[](const std::vector<size_t>& indices) const {
    return index(indices)[0];
  }

  size_t rank() const {
    return _shape.size();
  }

  const Shape& shape() const {
    return _shape;
  }

  size_t dim(ssize_t dim) const {
    if (dim < 0)
      dim = _shape.size() + dim;
    return _shape[dim];
  }

  size_t stride(ssize_t dim) const {
    if (dim < 0)
      dim = _shape.size() + dim;
    return _strides[dim];
  }

  size_t size() const {
    return _size;
  }

  size_t reserved_memory() const {
    return _allocated_size * sizeof (T);
  }

  bool empty() const {
    return _size == 0;
  }

  StorageView& clear() {
    _size = 0;
    _shape.clear();
    _strides.clear();
    return *this;
  }

  StorageView& release() {
    if (_own_data && _data != nullptr)
      delete [] _data;
    _data = nullptr;
    _allocated_size = 0;
    return clear();
  }

  StorageView& reserve(size_t size) {
    release();
    _data = new T[size];
    _own_data = true;
    _allocated_size = size;
    return *this;
  }

  StorageView& reshape(const Shape& new_shape) {
    assert(_size == size(new_shape));
    _shape = new_shape;
    _strides = strides(new_shape);
    return *this;
  }

  StorageView& resize(const Shape& new_shape) {
    if (new_shape.empty())
      return clear();
    size_t new_size = size(new_shape);
    if (new_size > _allocated_size)
      reserve(new_size);
    _size = new_size;
    return reshape(new_shape);
  }

  StorageView& resize(size_t dim, size_t new_size) {
    Shape new_shape(_shape);
    new_shape[dim] = new_size;
    return resize(new_shape);
  }

  StorageView& grow(size_t dim, size_t size) {
    return resize(dim, _shape[dim] + size);
  }

  StorageView& shrink(size_t dim, size_t size) {
    return resize(dim, _shape[dim] - size);
  }

  template <typename U>
  StorageView& resize_as(const StorageView<U>& other) {
    return resize(other._shape);
  }

  StorageView& assign(const StorageView& other) {
    resize_as(other);
    return copy_from(other);
  }

  StorageView& assign(StorageView&& other) {
    std::swap(_data, other._data);
    std::swap(_own_data, other._own_data);
    std::swap(_allocated_size, other._allocated_size);
    std::swap(_size, other._size);
    std::swap(_shape, other._shape);
    std::swap(_strides, other._strides);
    return *this;
  }

  StorageView& assign(T* data, const Shape& shape) {
    release();
    _data = data;
    _own_data = false;
    _allocated_size = size(shape);
    _size = _allocated_size;
    return reshape(shape);
  }

  StorageView& operator=(const StorageView& other) {
    return assign(other);
  }

  StorageView& operator=(StorageView&& other) {
    return assign(std::move(other));
  }

  StorageView& fill(T value) {
    std::fill_n(_data, _size, value);
    return *this;
  }

  StorageView& copy_from(const StorageView& other) {
    assert(_size == other._size);
    std::copy_n(other._data, other._size, _data);
    return *this;
  }

  StorageView& shallow_copy(StorageView& other) {
    return assign(other._data, other._shape);
  }

  StorageView& deep_copy(const StorageView& other) {
    return assign(other);
  }

  template <typename U>
  void cast_to(StorageView<U>& other) const {
    other.resize(_shape);
    for (size_t i = 0; i < _size; ++i)
      other[i] = static_cast<U>(_data[i]);
  }

  template <typename U>
  StorageView<U> cast() const {
    StorageView<U> other;
    cast_to(other);
    return other;
  }

  template <typename U>
  friend std::ostream& operator<<(std::ostream& os, const StorageView<U>& storage);

protected:
  T* _data = nullptr;
  bool _own_data = true;
  size_t _allocated_size = 0;
  size_t _size = 0;
  Shape _shape;
  std::vector<size_t> _strides;

  static size_t size(const Shape& shape) {
    if (shape.empty())
      return 0;
    size_t size = 1;
    for (auto dim : shape)
      size *= dim;
    return size;
  }

  static size_t stride(const Shape& shape, size_t dim) {
    if (shape.empty())
      return 0;
    size_t stride = 1;
    for (size_t i = shape.size() - 1; i > dim; --i)
      stride *= shape[i];
    return stride;
  }

  static std::vector<size_t> strides(const Shape& shape) {
    if (shape.empty())
      return std::vector<size_t>();
    std::vector<size_t> strides(shape.size(), 1);
    for (size_t d = 0; d < strides.size() - 1; ++d)
      strides[d] = stride(shape, d);
    return strides;
  }

};

template <typename T>
std::ostream& operator<<(std::ostream& os, const StorageView<T>& storage) {
  for (size_t i = 0; i < storage.size(); ++i) {
    if (i > 0)
      os << ' ';
    os << storage[i];
  }
  os << std::endl;
  os << '[' << typeid(T).name() << " storage viewed as ";
  for (size_t i = 0; i < storage.rank(); ++i) {
    if (i > 0)
      os << 'x';
    os << storage.dim(i);
  }
  os << ']';
  return os;
}
