#include "ctranslate2/storage_view.h"

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
#ifdef WITH_CUDA  // TODO: remove this CUDA specific guard.
    if (device == _device)
      return *this;
    StorageView device_copy(_shape, _dtype, device);
    if (device == Device::CUDA) {
      TYPE_DISPATCH(_dtype,
                    (cross_device_primitives<Device::CPU, Device::CUDA>::copy(
                      data<T>(), device_copy.data<T>(), _size)));
    } else {
      TYPE_DISPATCH(_dtype,
                    (cross_device_primitives<Device::CUDA, Device::CPU>::copy(
                      data<T>(), device_copy.data<T>(), _size)));
    }
    return device_copy;
#else
    device = device;
    return *this;
#endif
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
    _strides.clear();
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
    assert(_data != nullptr);
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
    return _strides[dim];
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
    assert(_size == size(new_shape));
    _shape = new_shape;
    _strides = strides(new_shape);
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
    resize_as(other);
    return copy_from(other);
  }

  StorageView& StorageView::assign(StorageView&& other) {
    assert(other._dtype == _dtype);
    assert(other._device == _device);
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
    TYPE_DISPATCH(_dtype, assign(other.data<T>(), other._shape));
    return *this;
  }

  StorageView& StorageView::deep_copy(const StorageView& other) {
    return assign(other);
  }

  StorageView& StorageView::copy_from(const StorageView& other) {
    TYPE_DISPATCH(other._dtype, copy_from(other.data<T>(), other._size));
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

  std::vector<size_t> StorageView::strides(const Shape& shape) {
    if (shape.empty())
      return std::vector<size_t>();
    std::vector<size_t> strides(shape.size(), 1);
    for (size_t d = 0; d < strides.size() - 1; ++d)
      strides[d] = stride(shape, d);
    return strides;
  }

  std::ostream& operator<<(std::ostream& os, const StorageView& storage) {
    if (storage.device() == Device::CPU) {
      TYPE_DISPATCH(
        storage.dtype(),
        // Do not spam output stream for large storages.
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
    }
    os << '[';
    if (storage.device() == Device::CUDA)
      os << "CUDA ";
    os << dtype_name(storage.dtype()) << " storage viewed as ";
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
    std::swap(a._strides, b._strides);
  }

}
