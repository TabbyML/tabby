#pragma once

#include <ostream>
#include <vector>

#include "types.h"
#include "utils.h"
#include "primitives/primitives.h"

namespace ctranslate2 {

#define ASSERT_DTYPE(DTYPE)                                             \
  if (_dtype != DTYPE) {                                                \
    THROW_INVALID_ARGUMENT("expected storage to be of type "            \
                           + dtype_name(DTYPE)                          \
                           + ", but is of type "                        \
                           + dtype_name(_dtype));                       \
  }

#define ASSERT_DEVICE(DEVICE)                                           \
  if (_device != DEVICE) {                                              \
    THROW_INVALID_ARGUMENT("expected storage to be on device "          \
                           + device_to_str(DEVICE)                      \
                           + ", but is on device "                      \
                           + device_to_str(_device));                   \
  }

#define ASSERT_COMPATIBLE(DTYPE, DEVICE)      \
  ASSERT_DTYPE(DTYPE);                        \
  ASSERT_DEVICE(DEVICE)

#define GUARD_DIM(DIM, RANK)                                            \
  do {                                                                  \
    if (DIM >= RANK)                                                    \
      THROW_INVALID_ARGUMENT("can't index dimension "                   \
                             + std::to_string(DIM)                      \
                             + " for a storage with rank "              \
                             + std::to_string(RANK));                   \
  } while (false)

  using Shape = std::vector<dim_t>;

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
    StorageView(DataType type = DataType::FLOAT, Device device = Device::CPU);
    StorageView(Device device, DataType type = DataType::FLOAT);

    // The reserved memory is uninitialized.
    StorageView(const Shape& shape, DataType type = DataType::FLOAT, Device device = Device::CPU);

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
      resize({});
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
    Device device() const {
      return _device;
    }

    StorageView to(Device D) const;

    // Actual storage type.
    DataType dtype() const {
      return _dtype;
    }

    // Allocated memory size.
    dim_t reserved_memory() const;
    // Clears the content (memory is still reserved).
    StorageView& clear();
    // Releases the memory.
    StorageView& release();
    // Reserves the memory.
    // If size is larger than the currently allocated size, a reallocation occurs and
    // the data is replaced by uninitialized memory.
    // If size is smaller than the currently allocated size, this a no-op.
    StorageView& reserve(dim_t size);
    bool owns_data() const;

    dim_t rank() const {
      return _shape.size();
    }

    const Shape& shape() const {
      return _shape;
    }

    dim_t dim(dim_t dim) const {
      if (dim < 0)
        dim = _shape.size() + dim;
      GUARD_DIM(dim, rank());
      return _shape[dim];
    }

    dim_t stride(dim_t dim) const {
      if (dim < 0)
        dim = _shape.size() + dim;
      GUARD_DIM(dim, rank());
      return compute_stride(_shape, dim);
    }

    dim_t size() const {
      return _size;
    }

    bool is_scalar() const {
      return _size == 1 && _shape.empty();
    }

    bool empty() const {
      return _size == 0;
    }

    operator bool() const {
      return !empty();
    }

    StorageView& reshape(const Shape& new_shape);

    // Resize methods (see also reserve() for information about reallocation policy).
    StorageView& resize_as(const StorageView& other);
    StorageView& resize(const Shape& new_shape);
    StorageView& resize(dim_t dim, dim_t new_size);
    StorageView& grow(dim_t dim, dim_t size);
    StorageView& shrink(dim_t dim, dim_t size);

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
      ASSERT_DTYPE(DataTypeToEnum<T>::value);
      return static_cast<T*>(_data);
    }

    template <typename T>
    const T* data() const {
      ASSERT_DTYPE(DataTypeToEnum<T>::value);
      return static_cast<const T*>(_data);
    }

    template <typename T>
    std::vector<T> to_vector() const {
      if (_device != Device::CPU)
        return to(Device::CPU).to_vector<T>();
      const T* begin = data<T>();
      const T* end = begin + _size;
      return std::vector<T>(begin, end);
    }

    template <typename T>
    T* index(const std::vector<dim_t>& indices) {
      return const_cast<T*>(static_cast<const StorageView&>(*this).index<T>(indices));
    }

    template <typename T>
    const T* index(const std::vector<dim_t>& indices) const {
      ASSERT_DTYPE(DataTypeToEnum<T>::value);
      dim_t offset = 0;
      for (size_t i = 0; i < indices.size(); ++i)
        offset += indices[i] * stride(i);
      if (offset >= _size)
        THROW_INVALID_ARGUMENT("computed index is out of bounds ("
                               + std::to_string(offset) + " >= "
                               + std::to_string(_size) + ")");
      return data<T>() + offset;
    }

    template <typename T>
    T& at(dim_t index) {
      return const_cast<T&>(static_cast<const StorageView&>(*this).at<T>(index));
    }

    template <typename T>
    const T& at(dim_t index) const {
      if (index >= _size)
        THROW_INVALID_ARGUMENT("index is out of bounds ("
                               + std::to_string(index) + " >= "
                               + std::to_string(_size) + ")");
      return data<T>()[index];
    }

    template <typename T>
    T& at(const std::vector<dim_t>& indices) {
      return index<T>(indices)[0];
    }

    template <typename T>
    const T& at(const std::vector<dim_t>& indices) const {
      return index<T>(indices)[0];
    }

    template <typename T>
    T as_scalar() const {
      if (!is_scalar())
        THROW_INVALID_ARGUMENT("storage is not a scalar: rank is "
                               + std::to_string(rank()) + " (expected 0) and size is "
                               + std::to_string(_size) + " (expected 1)");
      return scalar_at<T>({});
    }

    template <typename T>
    T scalar_at(const std::vector<dim_t>& indices) const;

    template <typename T>
    StorageView& view(T* data, const Shape& shape) {
      ASSERT_DTYPE(DataTypeToEnum<T>::value);
      release();
      _data = static_cast<void*>(data);
      _own_data = false;
      _allocated_size = compute_size(shape);
      _size = _allocated_size;
      return reshape(shape);
    }

    template <typename T>
    StorageView& fill(T value);

    StorageView& copy_from(const StorageView& other);

    template <typename T>
    StorageView& copy_from(const T* data, dim_t size, Device device);

    friend void swap(StorageView& a, StorageView& b);
    friend std::ostream& operator<<(std::ostream& os, const StorageView& storage);

  protected:
    DataType _dtype = DataType::FLOAT;
    Device _device = Device::CPU;
    int _device_index = 0;
    void* _data = nullptr;
    bool _own_data = true;
    dim_t _allocated_size = 0;
    dim_t _size = 0;
    Shape _shape;

    static dim_t compute_size(const Shape& shape);
    static dim_t compute_stride(const Shape& shape, dim_t dim);

  };

}
