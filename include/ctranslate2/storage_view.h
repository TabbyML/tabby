#pragma once

#include <ostream>
#include <vector>

#ifdef _MSC_VER
#  include <BaseTsd.h>
   typedef SSIZE_T ssize_t;
#endif

#include "types.h"
#include "primitives/primitives.h"

using Shape = std::vector<size_t>;

namespace ctranslate2 {

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
    StorageView(DataType type = DataType::DT_FLOAT, Device device = Device::CPU);
    StorageView(Device device, DataType type = DataType::DT_FLOAT);
    StorageView(const Shape& shape, DataType type = DataType::DT_FLOAT, Device device = Device::CPU);

    template <typename T>
    StorageView(const Shape& shape, T init = T(), Device device = Device::CPU)
      : _dtype(DataTypeToEnum<T>::value)
      , _device(device) {
      resize(shape);
      fill(init);
    }

    template <typename T>
    StorageView(T scalar, Device device = Device::CPU)
      : _dtype(DataTypeToEnum<T>::value)
      , _device(device) {
      resize({1});
      fill(scalar);
    }

    // Create from a std::vector (copy).
    template <typename T>
    StorageView(const Shape& shape, const std::vector<T>& init, Device device = Device::CPU)
      : _dtype(DataTypeToEnum<T>::value)
      , _device(device) {
      resize(shape);
      copy_from(init.data(), init.size(), Device::CPU);
    }

    // Create from a buffer (no copy).
    template <typename T>
    StorageView(const Shape& shape, T* data, Device device = Device::CPU)
      : _dtype(DataTypeToEnum<T>::value)
      , _device(device) {
      view(data, shape);
    }

    // Copy constructor.
    StorageView(const StorageView& other);
    // Move constructor (swap of each attribute).
    StorageView(StorageView&& other);
    ~StorageView();

    // Device management.
    Device device() const;
    StorageView to(Device D) const;

    // Actual storage type.
    DataType dtype() const;

    void assert_dtype(DataType dtype) const;
    void assert_device(Device device) const;
    void assert_compatible(DataType dtype, Device device) const;

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

    void* buffer();
    const void* buffer() const;

    template <typename T>
    T* data() {
      assert_dtype(DataTypeToEnum<T>::value);
      return reinterpret_cast<T*>(_data);
    }

    template <typename T>
    const T* data() const {
      assert_dtype(DataTypeToEnum<T>::value);
      return reinterpret_cast<const T*>(_data);
    }

    template <typename T>
    T* index(const std::vector<size_t>& indices) {
      return const_cast<T*>(static_cast<const StorageView&>(*this).index<T>(indices));
    }

    template <typename T>
    const T* index(const std::vector<size_t>& indices) const {
      assert_dtype(DataTypeToEnum<T>::value);
      size_t offset = 0;
      for (size_t i = 0; i < indices.size(); ++i)
        offset += indices[i] * stride(i);
      if (offset >= _size)
        throw std::invalid_argument("index: computed index is out of bounds");
      return data<T>() + offset;
    }

    template <typename T>
    T& at(size_t index) {
      return const_cast<T&>(static_cast<const StorageView&>(*this).at<T>(index));
    }

    template <typename T>
    const T& at(size_t index) const {
      if (index >= _size)
        throw std::invalid_argument("at: index is out of bounds");
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
    T as_scalar() const {
      if (!is_scalar())
        throw std::invalid_argument("not a scalar");
      return scalar_at<T>({0});
    }

    template <typename T>
    T scalar_at(const std::vector<size_t>& indices) const;

    template <typename T>
    StorageView& view(T* data, const Shape& shape) {
      assert_dtype(DataTypeToEnum<T>::value);
      release();
      _data = static_cast<void*>(data);
      _own_data = false;
      _allocated_size = size(shape);
      _size = _allocated_size;
      return reshape(shape);
    }

    template <typename T>
    StorageView& fill(T value);

    StorageView& copy_from(const StorageView& other);

    template <typename T>
    StorageView& copy_from(const T* data, size_t size, Device device);

    friend void swap(StorageView& a, StorageView& b);
    friend std::ostream& operator<<(std::ostream& os, const StorageView& storage);

  protected:
    DataType _dtype = DataType::DT_FLOAT;
    Device _device = Device::CPU;
    void* _data = nullptr;
    bool _own_data = true;
    size_t _allocated_size = 0;
    size_t _size = 0;
    Shape _shape;

    static size_t size(const Shape& shape);
    static size_t stride(const Shape& shape, size_t dim);

  };

}
