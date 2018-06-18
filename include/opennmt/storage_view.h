#pragma once

#include <cassert>
#include <ostream>
#include <vector>

#include "types.h"
#include "primitives/primitives.h"

using Shape = std::vector<size_t>;

namespace opennmt {

  // This class is a light wrapper around an allocated buffer which adds shape information.
  //
  // 1. it can be resized, reshaped, copied, and assigned;
  // 2. it can view an existing buffer to avoid memory copy;
  // 3. the buffer can be of any type and uses dynamic type dispatch (to allow collections
  //    of heterogeneous storages);
  // 4. allocation is aligned by default to 64 bytes.
  class StorageView
  {
  public:
    StorageView(DataType type = DataType::DT_FLOAT);
    StorageView(const Shape& shape, DataType type = DataType::DT_FLOAT);

    template <typename T>
    StorageView(const Shape& shape, T init = T())
      : _dtype(DataTypeToEnum<T>::value) {
      resize(shape);
      fill(init);
    }

    template <typename T>
    StorageView(T scalar)
      : _dtype(DataTypeToEnum<T>::value) {
      resize({1});
      fill(scalar);
    }

    // Create from a std::vector (copy).
    template <typename T>
    StorageView(const Shape& shape, const std::vector<T>& init)
      : _dtype(DataTypeToEnum<T>::value) {
      resize(shape);
      copy_from(init.data(), init.size());
    }

    // Create from a buffer (no copy).
    template <typename T>
    StorageView(const Shape& shape, T* data)
      : _dtype(DataTypeToEnum<T>::value) {
      assign(data, shape);
    }

    // Copy constructor.
    StorageView(const StorageView& other);
    // Move constructor (swap of each attribute).
    StorageView(StorageView&& other);
    ~StorageView();

    // Actual storage type.
    DataType dtype() const;
    // Allocated memory size.
    size_t reserved_memory() const;
    // Clears the content (memory is still reserved).
    StorageView& clear();
    // Releases the memory.
    StorageView& release();
    // Reserves this size (data are discarded).
    StorageView& reserve(size_t size);

    size_t rank() const;
    const Shape& shape() const;
    size_t dim(ssize_t dim) const;
    size_t stride(ssize_t dim) const;
    size_t size() const;
    bool is_scalar() const;
    bool empty() const;

    StorageView& reshape(const Shape& new_shape);

    StorageView& resize_as(const StorageView& other);
    StorageView& resize(const Shape& new_shape);
    StorageView& resize(size_t dim, size_t new_size);
    StorageView& grow(size_t dim, size_t size);
    StorageView& shrink(size_t dim, size_t size);

    // Assignment operators.
    StorageView& operator=(const StorageView& other);
    StorageView& operator=(StorageView&& other);
    StorageView& assign(const StorageView& other);
    StorageView& assign(StorageView&& other);

    StorageView& shallow_copy(StorageView& other);
    StorageView& deep_copy(const StorageView& other);

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

    template <typename T>
    StorageView& fill(T value) {
      assert(DataTypeToEnum<T>::value == _dtype);
      primitives::fill(data<T>(), value, _size);
      return *this;
    }

    StorageView& copy_from(const StorageView& other);

    template <typename T>
    StorageView& copy_from(const T* data, size_t size) {
      assert(DataTypeToEnum<T>::value == _dtype);
      assert(size == _size);
      primitives::copy(data, this->data<T>(), size);
      return *this;
    }

    friend void swap(StorageView& a, StorageView& b);
    friend std::ostream& operator<<(std::ostream& os, const StorageView& storage);

  protected:
    DataType _dtype = DataType::DT_FLOAT;
    void* _data = nullptr;
    bool _own_data = true;
    size_t _allocated_size = 0;
    size_t _size = 0;
    Shape _shape;
    std::vector<size_t> _strides;

    static size_t size(const Shape& shape);
    static size_t stride(const Shape& shape, size_t dim);
    static std::vector<size_t> strides(const Shape& shape);

  };

}
