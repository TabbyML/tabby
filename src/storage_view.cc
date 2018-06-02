#include "opennmt/storage_view.h"

#include "opennmt/utils.h"

#define ALIGNMENT 64

namespace opennmt {

  StorageView::StorageView(DataType type)
    : _dtype(type) {
  }

  StorageView::StorageView(const Shape& shape, DataType type)
    : _dtype(type) {
    resize(shape);
    TYPE_DISPATCH(type, fill(T()));
  }

  StorageView::StorageView(const StorageView& other)
    : _dtype(other._dtype) {
    assign(other);
  }

  StorageView::StorageView(StorageView&& other)
    : _dtype(other._dtype) {
    assign(std::move(other));
  }

  StorageView::~StorageView() {
    release();
  }

  DataType StorageView::dtype() const {
    return _dtype;
  }

  size_t StorageView::reserved_memory() const {
    size_t buffer_size;
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
    if (_own_data && _buffer != nullptr)
      free(_buffer);
    _data = nullptr;
    _buffer = nullptr;
    _allocated_size = 0;
    return clear();
  }

  StorageView& StorageView::reserve(size_t size) {
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
    std::swap(_data, other._data);
    std::swap(_own_data, other._own_data);
    std::swap(_allocated_size, other._allocated_size);
    std::swap(_size, other._size);
    std::swap(_shape, other._shape);
    std::swap(_strides, other._strides);
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
