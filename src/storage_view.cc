#include "ctranslate2/storage_view.h"

#include "ctranslate2/primitives.h"

#include "dispatch.h"

#define PRINT_MAX_VALUES 6

namespace ctranslate2 {

  StorageView::StorageView(DataType type, Device device)
    : _dtype(type)
    , _device(device)
    , _device_index(get_device_index(device)) {
  }

  StorageView::StorageView(Device device, DataType type)
    : _dtype(type)
    , _device(device)
    , _device_index(get_device_index(device)) {
  }

  StorageView::StorageView(Shape shape, DataType type, Device device)
    : _dtype(type)
    , _device(device)
    , _device_index(get_device_index(device)) {
    resize(std::move(shape));
  }

  template <typename T>
  StorageView::StorageView(Shape shape, T init, Device device)
    : _dtype(DataTypeToEnum<T>::value)
    , _device(device)
    , _device_index(get_device_index(device)) {
    resize(std::move(shape));
    fill(init);
  }

  template <typename T>
  StorageView::StorageView(T scalar, Device device)
    : _dtype(DataTypeToEnum<T>::value)
    , _device(device)
    , _device_index(get_device_index(device)) {
    resize({});
    fill(scalar);
  }

  template <typename T>
  StorageView::StorageView(Shape shape, const std::vector<T>& init, Device device)
    : _dtype(DataTypeToEnum<T>::value)
    , _device(device)
    , _device_index(get_device_index(device)) {
    resize(std::move(shape));
    copy_from(init.data(), init.size(), Device::CPU);
  }

  template <typename T>
  StorageView::StorageView(Shape shape, T* data, Device device)
    : _dtype(DataTypeToEnum<T>::value)
    , _device(device)
    , _device_index(get_device_index(device)) {
    view(data, std::move(shape));
  }

  StorageView::StorageView(const StorageView& other, bool synchronous)
    : _dtype(other._dtype)
    , _device(other._device)
    , _device_index(other._device_index) {
    ScopedDeviceSetter scoped_device_setter(_device, _device_index);
    copy_from(other, synchronous);
  }

  StorageView::StorageView(StorageView&& other) noexcept
    : _dtype(other._dtype)
    , _device(other._device)
    , _device_index(other._device_index)
    , _allocator(other._allocator)
    , _data(other._data)
    , _allocated_size(other._allocated_size)
    , _size(other._size)
    , _shape(std::move(other._shape)) {
    other._allocator = nullptr;  // other no longer owns the data.
    other.release();
  }

  StorageView::~StorageView() {
    release();
  }

  StorageView StorageView::to(Device device) const {
    StorageView device_copy(_shape, _dtype, device);
    return device_copy.copy_from(*this);
  }

  StorageView StorageView::to(DataType dtype) const {
    if (_dtype == dtype)
      return *this;
    StorageView converted(_shape, dtype, _device);
    if (_dtype == DataType::FLOAT32 && dtype == DataType::FLOAT16) {
      DEVICE_DISPATCH(_device,
                      primitives<D>::convert(data<float>(), converted.data<float16_t>(), _size));
    } else if (_dtype == DataType::FLOAT16 && dtype == DataType::FLOAT32) {
      DEVICE_DISPATCH(_device,
                      primitives<D>::convert(data<float16_t>(), converted.data<float>(), _size));
    } else if (_dtype == DataType::FLOAT32 && dtype == DataType::BFLOAT16) {
      DEVICE_DISPATCH(_device,
                      primitives<D>::convert(data<float>(), converted.data<bfloat16_t>(), _size));
    } else if (_dtype == DataType::BFLOAT16 && dtype == DataType::FLOAT32) {
      DEVICE_DISPATCH(_device,
                      primitives<D>::convert(data<bfloat16_t>(), converted.data<float>(), _size));
    } else if (_dtype == DataType::BFLOAT16 && dtype == DataType::FLOAT16) {
      DEVICE_DISPATCH(_device,
                      primitives<D>::convert(data<bfloat16_t>(), converted.data<float16_t>(), _size));
    } else if (_dtype == DataType::FLOAT16 && dtype == DataType::BFLOAT16) {
      DEVICE_DISPATCH(_device,
                      primitives<D>::convert(data<float16_t>(), converted.data<bfloat16_t>(), _size));
    } else {
      // TODO: support other conversions.
      throw std::invalid_argument("Conversion from " + dtype_name(_dtype)
                                  + " to " + dtype_name(dtype) + " is not yet implemented");
    }
    return converted;
  }

  StorageView StorageView::to_float16() const {
    return to(DataType::FLOAT16);
  }

  StorageView StorageView::to_float32() const {
    return to(DataType::FLOAT32);
  }

  StorageView& StorageView::move_to(Device device, DataType dtype) {
    if (_dtype != dtype)
      *this = to(dtype);
    if (_device != device)
      *this = to(device);
    return *this;
  }

  dim_t StorageView::reserved_memory() const {
    return _allocated_size * item_size();
  }

  StorageView& StorageView::clear() {
    _size = 0;
    _shape.clear();
    return *this;
  }

  StorageView& StorageView::release() {
    if (_allocator && _data != nullptr) {
      _allocator->free(_data, _device_index);
    }
    _data = nullptr;
    _allocator = nullptr;
    _allocated_size = 0;
    return clear();
  }

  StorageView& StorageView::reserve(dim_t size) {
    if (size <= _allocated_size)
      return *this;
    release();
    _allocator = &get_allocator(_device);
    _data = _allocator->allocate(size * item_size(), _device_index);
    if (_data == nullptr)
      THROW_RUNTIME_ERROR("failed to allocated memory");
    _allocated_size = size;
    return *this;
  }

  bool StorageView::owns_data() const {
    return _allocator;
  }

  dim_t StorageView::item_size() const {
    dim_t size = 0;
    TYPE_DISPATCH(_dtype, size = sizeof (T));
    return size;
  }

  StorageView& StorageView::reshape(Shape new_shape) {
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
      new_shape[unknown_dim] = _size / known_size;
    } else if (_size != known_size) {
      THROW_INVALID_ARGUMENT("new shape size (" + std::to_string(known_size)
                             + ") is incompatible with current size ("
                             + std::to_string(_size) + ")");
    }

    _shape = std::move(new_shape);
    return *this;
  }

  StorageView& StorageView::expand_dims(dim_t dim) {
    if (dim < 0)
      dim = _shape.size() + dim + 1;
    if (dim > rank())
      throw std::out_of_range("can't insert dimension at index " + std::to_string(dim));
    _shape.insert(_shape.begin() + dim, 1);
    return *this;
  }

  StorageView& StorageView::squeeze(dim_t dim) {
    if (dim < 0)
      dim = _shape.size() + dim;
    GUARD_DIM(dim, rank());
    if (_shape[dim] != 1)
      throw std::invalid_argument("dimension " + std::to_string(dim)
                                  + " has size " + std::to_string(_shape[dim])
                                  + " which can't be squeezed");
    _shape.erase(_shape.begin() + dim);
    return *this;
  }

  StorageView& StorageView::resize(Shape new_shape) {
    const dim_t new_size = compute_size(new_shape);
    reserve(new_size);
    _size = new_size;
    _shape = std::move(new_shape);
    return *this;
  }

  StorageView& StorageView::resize(dim_t dim, dim_t new_size) {
    GUARD_DIM(dim, rank());
    Shape new_shape(_shape);
    new_shape[dim] = new_size;
    return resize(std::move(new_shape));
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
    if (other.empty() && other.rank() == 0)
      return clear();
    return resize(other.shape());
  }

  StorageView& StorageView::operator=(const StorageView& other) {
    if (this != &other) {
      if (_device != other._device || _device_index != other._device_index)
        release();
      _device = other._device;
      _device_index = other._device_index;
      _dtype = other._dtype;
      copy_from(other);
    }
    return *this;
  }

  StorageView& StorageView::operator=(StorageView&& other) noexcept {
    std::swap(_dtype, other._dtype);
    std::swap(_device, other._device);
    std::swap(_device_index, other._device_index);
    std::swap(_allocator, other._allocator);
    std::swap(_data, other._data);
    std::swap(_allocated_size, other._allocated_size);
    std::swap(_size, other._size);
    std::swap(_shape, other._shape);
    return *this;
  }

  StorageView& StorageView::shallow_copy(StorageView& other) {
    _dtype = other._dtype;
    TYPE_DISPATCH(_dtype, view(other.data<T>(), other._shape));
    // Device info should be set after view(), which releases memory on the current device.
    _device = other._device;
    _device_index = other._device_index;
    return *this;
  }

  StorageView StorageView::sync_copy() const {
    return StorageView(*this, /*synchronous=*/true);
  }

  void* StorageView::buffer() {
    return _data;
  }

  const void* StorageView::buffer() const {
    return _data;
  }

  template <typename T>
  T* StorageView::data() {
    ASSERT_DTYPE(DataTypeToEnum<T>::value);
    return static_cast<T*>(_data);
  }

  template <typename T>
  const T* StorageView::data() const {
    ASSERT_DTYPE(DataTypeToEnum<T>::value);
    return static_cast<const T*>(_data);
  }

  template <typename T>
  std::vector<T> StorageView::to_vector() const {
    if (_device != Device::CPU)
      return to(Device::CPU).to_vector<T>();
    const T* begin = data<T>();
    const T* end = begin + _size;
    return std::vector<T>(begin, end);
  }

  template <typename T>
  T* StorageView::index(std::initializer_list<dim_t> indices) {
    return const_cast<T*>(static_cast<const StorageView&>(*this).index<T>(indices));
  }

  template <typename T>
  const T* StorageView::index(std::initializer_list<dim_t> indices) const {
    const dim_t num_indices = indices.size();
    if (num_indices != rank())
      THROW_INVALID_ARGUMENT("number of indexed dimensions ("
                             + std::to_string(indices.size())
                             + ") does not match the storage rank ("
                             + std::to_string(rank()) + ")");

    dim_t offset = 0;
    if (num_indices > 0) {
      dim_t stride = 1;
      auto index_it = std::crbegin(indices);
      auto dim_it = std::crbegin(_shape);
      for (; index_it != std::crend(indices); ++index_it, ++dim_it) {
        offset += *index_it * stride;
        stride *= *dim_it;
      }
    }

    if (offset >= _size)
      THROW_INVALID_ARGUMENT("computed index is out of bounds ("
                             + std::to_string(offset) + " >= "
                             + std::to_string(_size) + ")");
    return data<T>() + offset;
  }

  StorageView& StorageView::copy_from(const StorageView& other, bool synchronous) {
    resize_as(other);
    TYPE_DISPATCH(other._dtype, copy_from(other.data<T>(), other._size, other._device, synchronous));
    return *this;
  }

  template <typename U>
  U StorageView::scalar_at(std::initializer_list<dim_t> indices) const {
    auto scalar = U();
    DEVICE_AND_TYPE_DISPATCH(_device, _dtype, scalar = primitives<D>::at(index<T>(indices), 0));
    return scalar;
  }

  template <typename T>
  StorageView& StorageView::view(T* data, Shape shape) {
    ASSERT_DTYPE(DataTypeToEnum<T>::value);
    release();
    _data = static_cast<void*>(data);
    _allocated_size = compute_size(shape);
    _size = _allocated_size;
    return reshape(std::move(shape));
  }

  StorageView& StorageView::view(void* data, Shape shape) {
    TYPE_DISPATCH(_dtype, view(reinterpret_cast<T*>(data), std::move(shape)));
    return *this;
  }

  template <typename T>
  StorageView& StorageView::fill(T value) {
    DEVICE_DISPATCH(_device, primitives<D>::fill(data<T>(), value, _size));
    return *this;
  }

  StorageView& StorageView::zero() {
    DEVICE_AND_TYPE_DISPATCH(_device, _dtype, primitives<D>::fill(data<T>(), T(0), _size));
    return *this;
  }

  template <typename T>
  StorageView& StorageView::copy_from(const T* data, dim_t size, Device device, bool synchronous) {
    if (size != _size)
      THROW_INVALID_ARGUMENT("buffer to copy is of size " + std::to_string(size)
                             + " but current storage size is " + std::to_string(_size));
#ifdef CT2_WITH_CUDA
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

    if (synchronous)
      synchronize_stream(_device);

    return *this;
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
    os << "[" << device_to_str(storage.device(), storage.device_index())
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

#define DECLARE_IMPL(T)                                                 \
  template                                                              \
  StorageView::StorageView(Shape shape, T init, Device device);         \
  template StorageView::StorageView(T scalar, Device device);           \
  template                                                              \
  StorageView::StorageView(Shape shape,                                 \
                           const std::vector<T>& init,                  \
                           Device device);                              \
  template                                                              \
  StorageView::StorageView(Shape shape, T* data, Device device);        \
  template T* StorageView::data();                                      \
  template const T* StorageView::data() const;                          \
  template std::vector<T> StorageView::to_vector() const;               \
  template T* StorageView::index(std::initializer_list<dim_t> indices); \
  template const T*                                                     \
  StorageView::index(std::initializer_list<dim_t> indices) const;       \
  template T                                                            \
  StorageView::scalar_at(std::initializer_list<dim_t> indices) const;   \
  template StorageView& StorageView::view(T* data, Shape shape);        \
  template StorageView& StorageView::fill(T value);                     \
  template StorageView&                                                 \
  StorageView::copy_from(const T* data, dim_t size, Device device, bool);

  DECLARE_ALL_TYPES(DECLARE_IMPL)

}
