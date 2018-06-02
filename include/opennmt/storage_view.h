#pragma once

#include <cassert>
#include <ostream>
#include <stdexcept>
#include <vector>

#include "compute.h"
#include "opennmt/utils.h"
#include "opennmt/types.h"

#define ALIGNMENT 64

using Shape = std::vector<size_t>;

namespace opennmt {

  // The `StorageView` class is a light wrapper around an allocated buffer to give
  // it a sense of shape.
  //
  // 1. it can be resized, reshaped, copied, and assigned
  // 2. it can view an existing buffer to avoid memory copy
  // 3. the buffer can be of any type and casting is supported
  // 4. allocation is aligned by default to 64 bytes: wasted space is minimal and it
  //    is required when working with intrinsics up to AVX512
  class StorageView
  {
  public:
    StorageView(DataType type = DataType::DT_FLOAT)
      : _dtype(type) {
    }

    StorageView(const Shape& shape, DataType type = DataType::DT_FLOAT)
      : _dtype(type) {
      resize(shape);
      TYPE_DISPATCH(type, fill(T()));
    }

    template <typename T>
    StorageView(const Shape& shape, T init = T())
      : _dtype(DataTypeToEnum<T>::value) {
      resize(shape);
      fill(init);
    }

    template <typename T>
    StorageView(const Shape& shape, const std::vector<T>& init)
      : _dtype(DataTypeToEnum<T>::value) {
      resize(shape);
      copy_from(init.data(), init.size());
    }

    template <typename T>
    StorageView(T* data, const Shape& shape)
      : _dtype(DataTypeToEnum<T>::value) {
      assign(data, shape);
    }

    StorageView(const StorageView& other)
      : _dtype(other._dtype) {
      assign(other);
    }

    StorageView(StorageView&& other)
      : _dtype(other._dtype) {
      assign(std::move(other));
    }

    ~StorageView() {
      release();
    }

    DataType dtype() const {
      return _dtype;
    }

    template <typename T>
    T* data() {
      assert(DataTypeToEnum<T>::value == _dtype);
      return reinterpret_cast<T*>(_data);
    }

    template <typename T>
    const T* data() const {
      assert(DataTypeToEnum<T>::value == _dtype);
      return reinterpret_cast<const T*>(_data);
    }

    template <typename T>
    T* index(const std::vector<size_t>& indices) {
      return const_cast<T*>(static_cast<const StorageView&>(*this).index<T>(indices));
    }

    template <typename T>
    const T* index(const std::vector<size_t>& indices) const {
      assert(DataTypeToEnum<T>::value == _dtype);
      size_t offset = 0;
      for (size_t i = 0; i < indices.size(); ++i)
        offset += indices[i] * _strides[i];
      assert(offset < _size);
      return data<T>() + offset;
    }

    template <typename T>
    T& at(size_t index) {
      return const_cast<T&>(static_cast<const StorageView&>(*this).at<T>(index));
    }

    template <typename T>
    const T& at(size_t index) const {
      assert(index < _size);
      return data<T>()[index];
    }

    template <typename T>
    T& at(const std::vector<size_t>& indices) {
      return index<T>(indices)[0];
    }

    template <typename T>
    const T& at(const std::vector<size_t>& indices) const {
      return index<T>(indices)[0];
    }

    size_t reserved_memory() const {
      size_t buffer_size;
      TYPE_DISPATCH(_dtype, buffer_size = _allocated_size * sizeof (T));
      return buffer_size;
    }

    StorageView& clear() {
      _size = 0;
      _shape.clear();
      _strides.clear();
      return *this;
    }

    StorageView& release() {
      if (_own_data && _buffer != nullptr)
        free(_buffer);
      _data = nullptr;
      _buffer = nullptr;
      _allocated_size = 0;
      return clear();
    }

    StorageView& reserve(size_t size) {
      release();
      size_t required_bytes = 0;
      TYPE_DISPATCH(_dtype, required_bytes = size * sizeof (T));
      size_t buffer_space = required_bytes + ALIGNMENT;
      _buffer = malloc(buffer_space);
      _data = _buffer;
      align(ALIGNMENT, required_bytes, _data, buffer_space);
      assert(_data != nullptr);
      _own_data = true;
      _allocated_size = size;
      return *this;
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

    bool empty() const {
      return _size == 0;
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

    StorageView& resize_as(const StorageView& other) {
      return resize(other.shape());
    }

    StorageView& assign(const StorageView& other) {
      resize_as(other);
      return copy_from(other);
    }

    StorageView& assign(StorageView&& other) {
      assert(other._dtype == _dtype);
      std::swap(_data, other._data);
      std::swap(_own_data, other._own_data);
      std::swap(_allocated_size, other._allocated_size);
      std::swap(_size, other._size);
      std::swap(_shape, other._shape);
      std::swap(_strides, other._strides);
      return *this;
    }

    template <typename T>
    StorageView& assign(T* data, const Shape& shape) {
      assert(DataTypeToEnum<T>::value == _dtype);
      release();
      _data = static_cast<void*>(data);
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

    StorageView& shallow_copy(StorageView& other) {
      TYPE_DISPATCH(_dtype, assign(other.data<T>(), other._shape));
      return *this;
    }

    StorageView& deep_copy(const StorageView& other) {
      return assign(other);
    }

    template <typename T>
    StorageView& fill(T value) {
      assert(DataTypeToEnum<T>::value == _dtype);
      compute::fill(data<T>(), value, _size);
      return *this;
    }

    StorageView& copy_from(const StorageView& other) {
      TYPE_DISPATCH(other._dtype, copy_from(other.data<T>(), other._size));
      return *this;
    }

    template <typename T>
    StorageView& copy_from(const T* data, size_t size) {
      assert(DataTypeToEnum<T>::value == _dtype);
      assert(size == _size);
      compute::copy(data, this->data<T>(), size);
      return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const StorageView& storage);

  protected:
    DataType _dtype = DataType::DT_FLOAT;
    void* _data = nullptr;
    void* _buffer = nullptr;
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


  std::ostream& operator<<(std::ostream& os, const opennmt::StorageView& storage) {
    TYPE_DISPATCH(
      storage.dtype(),
      if (storage.size() < 7) {
        for (size_t i = 0; i < storage.size(); ++i) {
          os << ' ' << storage.data<T>()[i];
        }
      } else {
        os << " " << storage.data<T>()[0]
           << " " << storage.data<T>()[1]
           << " " << storage.data<T>()[2]
           << " ..."
           << " " << storage.data<T>()[storage.size() - 3]
           << " " << storage.data<T>()[storage.size() - 2]
           << " " << storage.data<T>()[storage.size() - 1];
      }
      os << std::endl);
    os << '[' << dtype_name(storage.dtype()) << " storage viewed as ";
    for (size_t i = 0; i < storage.rank(); ++i) {
      if (i > 0)
        os << 'x';
      os << storage.dim(i);
    }
    os << ']';
    return os;
  }

}
