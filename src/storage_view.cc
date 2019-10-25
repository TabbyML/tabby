#include "ctranslate2/storage_view.h"

#include "./device_dispatch.h"

#define PRINT_MAX_VALUES 6

namespace ctranslate2 {

  StorageView::StorageView(DataType type, Device device)
    : _dtype(type)
    , _device(device) {
  }

  StorageView::StorageView(Device device, DataType type)
    : _dtype(type)
    , _device(device) {
  }

  StorageView::StorageView(const Shape& shape, DataType type, Device device)
    : _dtype(type)
    , _device(device) {
    resize(shape);
    TYPE_DISPATCH(type, fill(T()));
  }

  StorageView::StorageView(const StorageView& other)
    : _dtype(other._dtype)
    , _device(other._device) {
    assign(other);
  }

  StorageView::StorageView(StorageView&& other)
    : _dtype(other._dtype)
    , _device(other._device) {
    assign(std::move(other));
  }

  StorageView::~StorageView() {
    release();
  }

  Device StorageView::device() const {
    return _device;
  }

  StorageView StorageView::to(Device device) const {
    StorageView device_copy(_shape, _dtype, device);
    return device_copy.copy_from(*this);
  }

  DataType StorageView::dtype() const {
    return _dtype;
  }

  size_t StorageView::reserved_memory() const {
    size_t buffer_size = 0;
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
      DEVICE_DISPATCH(_device, primitives<D>::free_data(_data));
    }
    _data = nullptr;
    _allocated_size = 0;
    return clear();
  }

  StorageView& StorageView::reserve(size_t size) {
    release();
    size_t required_bytes = 0;
    TYPE_DISPATCH(_dtype, required_bytes = size * sizeof (T));
    DEVICE_DISPATCH(_device, _data = primitives<D>::alloc_data(required_bytes));
    if (_data == nullptr)
      THROW_RUNTIME_ERROR("failed to allocated memory");
    _own_data = true;
    _allocated_size = size;
    return *this;
  }

  size_t StorageView::rank() const {
    return _shape.size();
  }

  const Shape& StorageView::shape() const {
    return _shape;
  }

  size_t StorageView::dim(ssize_t dim) const {
    if (dim < 0)
      dim = _shape.size() + dim;
    return _shape[dim];
  }

  size_t StorageView::stride(ssize_t dim) const {
    if (dim < 0)
      dim = _shape.size() + dim;
    return stride(_shape, dim);
  }

  size_t StorageView::size() const {
    return _size;
  }

  bool StorageView::is_scalar() const {
    return rank() == 1 && _size == 1;
  }

  bool StorageView::empty() const {
    return _size == 0;
  }

  StorageView& StorageView::reshape(const Shape& new_shape) {
    if (_size != size(new_shape))
      THROW_INVALID_ARGUMENT("new shape size (" + std::to_string(size(new_shape))
                             + ") is incompatible with current size (" + std::to_string(_size) + ")");
    _shape = new_shape;
    return *this;
  }

  StorageView& StorageView::resize(const Shape& new_shape) {
    if (new_shape.empty())
      return clear();
    size_t new_size = size(new_shape);
    if (new_size > _allocated_size)
      reserve(new_size);
    _size = new_size;
    return reshape(new_shape);
  }

  StorageView& StorageView::resize(size_t dim, size_t new_size) {
    Shape new_shape(_shape);
    new_shape[dim] = new_size;
    return resize(new_shape);
  }

  StorageView& StorageView::grow(size_t dim, size_t size) {
    return resize(dim, _shape[dim] + size);
  }

  StorageView& StorageView::shrink(size_t dim, size_t size) {
    return resize(dim, _shape[dim] - size);
  }

  StorageView& StorageView::resize_as(const StorageView& other) {
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
  T StorageView::scalar_at(const std::vector<size_t>& indices) const {
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
  StorageView& StorageView::copy_from(const T* data, size_t size, Device device) {
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

  size_t StorageView::size(const Shape& shape) {
    if (shape.empty())
      return 0;
    size_t size = 1;
    for (auto dim : shape)
      size *= dim;
    return size;
  }

  size_t StorageView::stride(const Shape& shape, size_t dim) {
    if (shape.empty())
      return 0;
    size_t stride = 1;
    for (size_t i = shape.size() - 1; i > dim; --i)
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
        for (size_t i = 0; i < printable.size(); ++i) {
          os << ' ';
          print_value(os, values[i]);
        }
      }
      else {
        for (size_t i = 0; i < PRINT_MAX_VALUES / 2; ++i) {
          os << ' ';
          print_value(os, values[i]);
        }
        os << " ...";
        for (size_t i = printable.size() - (PRINT_MAX_VALUES / 2); i < printable.size(); ++i) {
          os << ' ';
          print_value(os, values[i]);
        }
      }
      os << std::endl);
    os << "[" << device_to_str(storage.device())
       << " " << dtype_name(storage.dtype()) << " storage viewed as ";
    for (size_t i = 0; i < storage.rank(); ++i) {
      if (i > 0)
        os << 'x';
      os << storage.dim(i);
    }
    os << ']';
    return os;
  }

  void swap(StorageView& a, StorageView& b) {
    std::swap(a._dtype, b._dtype);
    std::swap(a._device, b._device);
    std::swap(a._data, b._data);
    std::swap(a._own_data, b._own_data);
    std::swap(a._allocated_size, b._allocated_size);
    std::swap(a._size, b._size);
    std::swap(a._shape, b._shape);
  }

#define DECLARE_IMPL(T)                                                 \
  template T                                                            \
  StorageView::scalar_at(const std::vector<size_t>& indices) const;     \
  template StorageView& StorageView::fill(T value);                     \
  template StorageView&                                                 \
  StorageView::copy_from(const T* data, size_t size, Device device);

  DECLARE_ALL_TYPES(DECLARE_IMPL)

}
