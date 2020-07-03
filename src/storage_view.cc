#include "ctranslate2/storage_view.h"

#include "./device_dispatch.h"

#define PRINT_MAX_VALUES 6

namespace ctranslate2 {

  StorageView::StorageView(DataType type, Device device)
    : _dtype(type)
    , _device(device) {
    DEVICE_DISPATCH(device, _device_index = primitives<D>::get_device());
  }

  StorageView::StorageView(Device device, DataType type)
    : _dtype(type)
    , _device(device) {
    DEVICE_DISPATCH(device, _device_index = primitives<D>::get_device());
  }

  StorageView::StorageView(const Shape& shape, DataType type, Device device)
    : _dtype(type)
    , _device(device) {
    DEVICE_DISPATCH(device, _device_index = primitives<D>::get_device());
    resize(shape);
  }

  StorageView::StorageView(const StorageView& other)
    : _dtype(other._dtype)
    , _device(other._device)
    , _device_index(other._device_index) {
    assign(other);
  }

  StorageView::StorageView(StorageView&& other)
    : _dtype(other._dtype)
    , _device(other._device)
    , _device_index(other._device_index) {
    assign(std::move(other));
  }

  StorageView::~StorageView() {
    release();
  }

  StorageView StorageView::to(Device device) const {
    StorageView device_copy(_shape, _dtype, device);
    return device_copy.copy_from(*this);
  }

  dim_t StorageView::reserved_memory() const {
    dim_t buffer_size = 0;
    TYPE_DISPATCH(_dtype, buffer_size = _allocated_size * sizeof (T));
    return buffer_size;
  }

  StorageView& StorageView::clear() {
    _size = 0;
    _shape.clear();
    return *this;
  }

  StorageView& StorageView::release() {
    if (_own_data && _data != nullptr) {
      DEVICE_DISPATCH(_device, primitives<D>::free_data(_data, _device_index));
    }
    _data = nullptr;
    _allocated_size = 0;
    return clear();
  }

  StorageView& StorageView::reserve(dim_t size) {
    if (size <= _allocated_size)
      return *this;
    release();
    dim_t required_bytes = 0;
    TYPE_DISPATCH(_dtype, required_bytes = size * sizeof (T));
    DEVICE_DISPATCH(_device, _data = primitives<D>::alloc_data(required_bytes, _device_index));
    if (_data == nullptr)
      THROW_RUNTIME_ERROR("failed to allocated memory");
    _own_data = true;
    _allocated_size = size;
    return *this;
  }

  bool StorageView::owns_data() const {
    return _own_data;
  }

  StorageView& StorageView::reshape(const Shape& new_shape) {
    dim_t unknown_dim = -1;
    dim_t known_size = 1;

    for (size_t i = 0; i < new_shape.size(); ++i) {
      const dim_t dim = new_shape[i];

      if (dim >= 0) {
        known_size *= dim;
      } else if (dim == -1) {
        if (unknown_dim >= 0)
          THROW_INVALID_ARGUMENT("only one dimension can be set to -1, got -1 for dimensions "
                                 + std::to_string(unknown_dim) + " and " + std::to_string(i));
        unknown_dim = i;
      } else {
        THROW_INVALID_ARGUMENT("invalid value " + std::to_string(dim)
                               + " for dimension " + std::to_string(i));
      }
    }

    if (unknown_dim >= 0) {
      if (_size % known_size != 0)
        THROW_INVALID_ARGUMENT("current size (" + std::to_string(_size)
                               + ") is not divisible by the known size ("
                               + std::to_string(known_size) + ")");
      Shape new_shape_copy(new_shape);
      new_shape_copy[unknown_dim] = _size / known_size;
      _shape = std::move(new_shape_copy);
    } else {
      if (_size != known_size)
        THROW_INVALID_ARGUMENT("new shape size (" + std::to_string(known_size)
                               + ") is incompatible with current size ("
                               + std::to_string(_size) + ")");
      _shape = new_shape;
    }

    return *this;
  }

  StorageView& StorageView::resize(const Shape& new_shape) {
    const dim_t new_size = compute_size(new_shape);
    reserve(new_size);
    _size = new_size;
    _shape = new_shape;
    return *this;
  }

  StorageView& StorageView::resize(dim_t dim, dim_t new_size) {
    GUARD_DIM(dim, rank());
    Shape new_shape(_shape);
    new_shape[dim] = new_size;
    return resize(new_shape);
  }

  StorageView& StorageView::grow(dim_t dim, dim_t size) {
    GUARD_DIM(dim, rank());
    return resize(dim, _shape[dim] + size);
  }

  StorageView& StorageView::shrink(dim_t dim, dim_t size) {
    GUARD_DIM(dim, rank());
    return resize(dim, _shape[dim] - size);
  }

  StorageView& StorageView::resize_as(const StorageView& other) {
    if (other.empty())
      return clear();
    return resize(other.shape());
  }

  StorageView& StorageView::assign(const StorageView& other) {
    ASSERT_COMPATIBLE(other._dtype, other._device);
    return copy_from(other);
  }

  StorageView& StorageView::assign(StorageView&& other) {
    ASSERT_COMPATIBLE(other._dtype, other._device);
    swap(*this, other);
    return *this;
  }

  StorageView& StorageView::operator=(const StorageView& other) {
    return assign(other);
  }

  StorageView& StorageView::operator=(StorageView&& other) {
    return assign(std::move(other));
  }

  StorageView& StorageView::shallow_copy(StorageView& other) {
    ASSERT_DEVICE(other._device);
    TYPE_DISPATCH(_dtype, view(other.data<T>(), other._shape));
    return *this;
  }

  StorageView& StorageView::deep_copy(const StorageView& other) {
    return assign(other);
  }

  void* StorageView::buffer() {
    return _data;
  }

  const void* StorageView::buffer() const {
    return _data;
  }

  StorageView& StorageView::copy_from(const StorageView& other) {
    resize_as(other);
    TYPE_DISPATCH(other._dtype, copy_from(other.data<T>(), other._size, other._device));
    return *this;
  }

  template <typename T>
  T StorageView::scalar_at(const std::vector<dim_t>& indices) const {
    T scalar = T();
    DEVICE_DISPATCH(_device, scalar = primitives<D>::deref(index<T>(indices), 0));
    return scalar;
  }

  template <typename T>
  StorageView& StorageView::fill(T value) {
    ASSERT_DTYPE(DataTypeToEnum<T>::value);
    DEVICE_DISPATCH(_device, primitives<D>::fill(data<T>(), value, _size));
    return *this;
  }

  template <typename T>
  StorageView& StorageView::copy_from(const T* data, dim_t size, Device device) {
    ASSERT_DTYPE(DataTypeToEnum<T>::value);
    if (size != _size)
      THROW_INVALID_ARGUMENT("buffer to copy is of size " + std::to_string(size)
                             + " but current storage size is " + std::to_string(_size));
#ifdef WITH_CUDA
    if (device != _device) {
      if (device == Device::CUDA)
        cross_device_primitives<Device::CUDA, Device::CPU>::copy(data, this->data<T>(), size);
      else
        cross_device_primitives<Device::CPU, Device::CUDA>::copy(data, this->data<T>(), size);
    } else
#endif
    {
      DEVICE_DISPATCH(device, primitives<D>::copy(data, this->data<T>(), size));
    }
    return *this;
  }

  dim_t StorageView::compute_size(const Shape& shape) {
    dim_t size = 1;
    for (const dim_t dim : shape)
      size *= dim;
    return size;
  }

  dim_t StorageView::compute_stride(const Shape& shape, dim_t dim) {
    GUARD_DIM(dim, static_cast<dim_t>(shape.size()));
    dim_t stride = 1;
    for (dim_t i = shape.size() - 1; i > dim; --i)
      stride *= shape[i];
    return stride;
  }

  template <typename T>
  std::ostream& print_value(std::ostream& os, const T& val) {
    os << val;
    return os;
  }

  template<>
  std::ostream& print_value(std::ostream& os, const int8_t& val) {
    os << static_cast<int32_t>(val);
    return os;
  }

  std::ostream& operator<<(std::ostream& os, const StorageView& storage) {
    StorageView printable(storage.dtype());
    printable.copy_from(storage);
    TYPE_DISPATCH(
      printable.dtype(),
      const auto* values = printable.data<T>();
      if (printable.size() <= PRINT_MAX_VALUES) {
        for (dim_t i = 0; i < printable.size(); ++i) {
          os << ' ';
          print_value(os, values[i]);
        }
      }
      else {
        for (dim_t i = 0; i < PRINT_MAX_VALUES / 2; ++i) {
          os << ' ';
          print_value(os, values[i]);
        }
        os << " ...";
        for (dim_t i = printable.size() - (PRINT_MAX_VALUES / 2); i < printable.size(); ++i) {
          os << ' ';
          print_value(os, values[i]);
        }
      }
      os << std::endl);
    os << "[" << device_to_str(storage.device()) << ':' << storage._device_index
       << " " << dtype_name(storage.dtype()) << " storage viewed as ";
    if (storage.is_scalar())
      os << "scalar";
    else {
      for (dim_t i = 0; i < storage.rank(); ++i) {
        if (i > 0)
          os << 'x';
        os << storage.dim(i);
      }
    }
    os << ']';
    return os;
  }

  void swap(StorageView& a, StorageView& b) {
    std::swap(a._dtype, b._dtype);
    std::swap(a._device, b._device);
    std::swap(a._device_index, b._device_index);
    std::swap(a._data, b._data);
    std::swap(a._own_data, b._own_data);
    std::swap(a._allocated_size, b._allocated_size);
    std::swap(a._size, b._size);
    std::swap(a._shape, b._shape);
  }

#define DECLARE_IMPL(T)                                                 \
  template T                                                            \
  StorageView::scalar_at(const std::vector<dim_t>& indices) const;      \
  template StorageView& StorageView::fill(T value);                     \
  template StorageView&                                                 \
  StorageView::copy_from(const T* data, dim_t size, Device device);

  DECLARE_ALL_TYPES(DECLARE_IMPL)

}
