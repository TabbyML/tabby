#pragma once

#include <ostream>
#include <vector>

#include "allocator.h"
#include "types.h"
#include "utils.h"

namespace ctranslate2 {

#define ASSERT_DTYPE(DTYPE)                                             \
  if (_dtype != DTYPE) {                                                \
    THROW_INVALID_ARGUMENT("expected storage to be of type "            \
                           + dtype_name(DTYPE)                          \
                           + ", but is of type "                        \
                           + dtype_name(_dtype));                       \
  }

#define GUARD_DIM(DIM, RANK)                                            \
  do {                                                                  \
    if (DIM >= RANK)                                                    \
      THROW_INVALID_ARGUMENT("can't index dimension "                   \
                             + std::to_string(DIM)                      \
                             + " for a storage with rank "              \
                             + std::to_string(RANK));                   \
  } while (false)

  using Shape = std::vector<dim_t>;

  inline dim_t compute_size(const Shape& shape) {
    dim_t size = 1;
    for (const dim_t dim : shape)
      size *= dim;
    return size;
  }

  inline dim_t compute_stride(const Shape& shape, dim_t dim) {
    const dim_t rank = shape.size();
    dim_t stride = 1;
    for (dim_t i = rank - 1; i > dim; --i)
      stride *= shape[i];
    return stride;
  }

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
    StorageView(DataType type = DataType::FLOAT32, Device device = Device::CPU);
    StorageView(Device device, DataType type = DataType::FLOAT32);

    // The reserved memory is uninitialized.
    StorageView(Shape shape, DataType type = DataType::FLOAT32, Device device = Device::CPU);

    template <typename T>
    StorageView(Shape shape, T init = T(), Device device = Device::CPU);

    template <typename T>
    explicit StorageView(T scalar, Device device = Device::CPU);

    // Create from a std::vector (copy).
    template <typename T>
    StorageView(Shape shape, const std::vector<T>& init, Device device = Device::CPU);

    // Create from a buffer (no copy).
    template <typename T>
    StorageView(Shape shape, T* data, Device device = Device::CPU);

    // Copy constructor.
    StorageView(const StorageView& other, bool synchronous = false);
    // Move constructor (swap of each attribute).
    StorageView(StorageView&& other) noexcept;
    ~StorageView();

    // Device management.
    Device device() const {
      return _device;
    }

    int device_index() const {
      return _device_index;
    }

    StorageView to(Device D) const;
    StorageView to(DataType dtype) const;
    StorageView to_float16() const;
    StorageView to_float32() const;

    StorageView& move_to(Device device, DataType dtype);

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

    dim_t item_size() const;

    bool is_scalar() const {
      return _size == 1 && _shape.empty();
    }

    bool empty() const {
      return _size == 0;
    }

    operator bool() const {
      return !empty();
    }

    StorageView& reshape(Shape new_shape);
    StorageView& expand_dims(dim_t dim);
    StorageView& squeeze(dim_t dim);

    // Resize methods (see also reserve() for information about reallocation policy).
    StorageView& resize_as(const StorageView& other);
    StorageView& resize(Shape new_shape);
    StorageView& resize(dim_t dim, dim_t new_size);
    StorageView& grow(dim_t dim, dim_t size);
    StorageView& shrink(dim_t dim, dim_t size);

    // Assignment operators.
    StorageView& operator=(const StorageView& other);
    StorageView& operator=(StorageView&& other) noexcept;

    StorageView& shallow_copy(StorageView& other);
    StorageView sync_copy() const;

    void* buffer();
    const void* buffer() const;

    template <typename T>
    T* data();
    template <typename T>
    const T* data() const;

    template <typename T>
    std::vector<T> to_vector() const;

    template <typename T>
    T* index(std::initializer_list<dim_t> indices);
    template <typename T>
    const T* index(std::initializer_list<dim_t> indices) const;

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
    T& at(std::initializer_list<dim_t> indices) {
      return index<T>(indices)[0];
    }

    template <typename T>
    const T& at(std::initializer_list<dim_t> indices) const {
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
    T scalar_at(std::initializer_list<dim_t> indices) const;

    template <typename T>
    StorageView& view(T* data, Shape shape);
    StorageView& view(void* data, Shape shape);

    template <typename T>
    StorageView& fill(T value);
    StorageView& zero();

    StorageView& copy_from(const StorageView& other, bool synchronous = false);

    template <typename T>
    StorageView& copy_from(const T* data, dim_t size, Device device, bool synchronous = false);

    friend std::ostream& operator<<(std::ostream& os, const StorageView& storage);

  protected:
    DataType _dtype = DataType::FLOAT32;
    Device _device = Device::CPU;
    int _device_index = 0;
    Allocator* _allocator = nullptr;
    void* _data = nullptr;
    dim_t _allocated_size = 0;
    dim_t _size = 0;
    Shape _shape;
  };

}
