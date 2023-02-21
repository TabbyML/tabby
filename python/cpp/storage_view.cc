#include "module.h"
#include "storage_view.h"

#include <sstream>

#include <ctranslate2/storage_view.h>

#include "utils.h"

using namespace pybind11::literals;

namespace ctranslate2 {
  namespace python {

    static DataType typestr_to_dtype(const std::string& typestr) {
      const auto type_code = typestr[1];
      const auto num_bytes = typestr[2];

      if (type_code == 'i') {
        if (num_bytes == '1')
          return DataType::INT8;
        if (num_bytes == '2')
          return DataType::INT16;
        if (num_bytes == '4')
          return DataType::INT32;

      } else if (type_code == 'f') {
        if (num_bytes == '2')
          return DataType::FLOAT16;
        if (num_bytes == '4')
          return DataType::FLOAT32;
      }

      throw std::invalid_argument("Unsupported type: " + typestr);
    }

    static std::string dtype_to_typestr(const DataType dtype) {
      // Assume little-endian.

      switch (dtype) {
      case DataType::FLOAT32:
        return "<f4";
      case DataType::FLOAT16:
        return "<f2";
      case DataType::INT8:
        return "|i1";
      case DataType::INT16:
        return "<i2";
      case DataType::INT32:
        return "<i4";
      default:
        return "";
      }
    }

    static StorageView create_view_from_array(py::object array) {
      auto device = Device::CPU;

      py::object interface_obj = py::getattr(array, "__array_interface__", py::none());
      if (interface_obj.is_none()) {
        interface_obj = py::getattr(array, "__cuda_array_interface__", py::none());
        if (interface_obj.is_none())
          throw std::invalid_argument("Object does not implement the array interface");
        device = Device::CUDA;
      }

      py::dict interface = interface_obj.cast<py::dict>();
      if (interface_obj.contains("strides") && !interface_obj["strides"].is_none())
        throw std::invalid_argument("StorageView does not support arrays with non contiguous memory");

      auto shape = interface["shape"].cast<Shape>();
      auto dtype = typestr_to_dtype(interface["typestr"].cast<std::string>());
      auto data = interface["data"].cast<py::tuple>();
      auto ptr = data[0].cast<uintptr_t>();
      auto read_only = data[1].cast<bool>();

      if (read_only)
        throw std::invalid_argument("StorageView does not support read-only arrays");

      StorageView view(dtype, device);
      view.view((void*)ptr, std::move(shape));
      return view;
    }

    static py::dict get_array_interface(const StorageView& view) {
      py::tuple shape(view.rank());
      for (size_t i = 0; i < shape.size(); ++i)
        shape[i] = view.dim(i);

      return py::dict(
        "shape"_a=shape,
        "typestr"_a=dtype_to_typestr(view.dtype()),
        "data"_a=py::make_tuple((uintptr_t)view.buffer(), false),
        "version"_a=3);
    }

    StorageViewWrapper::StorageViewWrapper(StorageView view)
      : _data_owner(py::none())
      , _view(std::move(view))
    {
    }

    StorageViewWrapper StorageViewWrapper::from_array(py::object array) {
      StorageViewWrapper view = create_view_from_array(array);
      view.set_data_owner(array);
      return view;
    }

    py::dict StorageViewWrapper::array_interface() const {
      if (_view.device() == Device::CUDA)
        throw pybind11::attribute_error("Cannot get __array_interface__ when the StorageView "
                                        "is viewing a CUDA array");
      return get_array_interface(_view);
    }

    py::dict StorageViewWrapper::cuda_array_interface() const {
      if (_view.device() == Device::CPU)
        throw pybind11::attribute_error("Cannot get __cuda_array_interface__ when the StorageView "
                                        "is viewing a CPU array");
      return get_array_interface(_view);
    }

    std::string StorageViewWrapper::str() const {
      std::ostringstream stream;
      stream << _view;
      return stream.str();
    }

    void register_storage_view(py::module& m) {
      py::class_<StorageViewWrapper>(
        m, "StorageView",
        R"pbdoc(
            An allocated buffer with shape information.

            The object implements the
            `Array Interface <https://numpy.org/doc/stable/reference/arrays.interface.html>`_
            and the
            `CUDA Array Interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_
            so that it can be passed to Numpy or PyTorch without copy.

            Example:

                >>> x = np.ones((2, 4), dtype=np.int32)
                >>> y = ctranslate2.StorageView.from_array(x)
                >>> print(y)
                 1 1 1 ... 1 1 1
                [cpu:0 int32 storage viewed as 2x4]
                >>> z = np.array(y)
                ...
                >>> x = torch.ones((2, 4), dtype=torch.int32, device="cuda")
                >>> y = ctranslate2.StorageView.from_array(x)
                >>> print(y)
                 1 1 1 ... 1 1 1
                [cuda:0 int32 storage viewed as 2x4]
                >>> z = torch.as_tensor(y, device="cuda")

        )pbdoc")

        .def_static("from_array", &StorageViewWrapper::from_array, py::arg("array"),
                    R"pbdoc(
                        Creates a ``StorageView`` from an object implementing the array interface.

                        Arguments:
                          array: An object implementing the array interface (e.g. a Numpy array
                            or a PyTorch Tensor).

                        Returns:
                          A new ``StorageView`` instance sharing the same data as the input array.

                        Raises:
                          ValueError: if the object does not implement the array interface or
                            uses an unsupported array specification.
                    )pbdoc")

        .def_property_readonly("__array_interface__", &StorageViewWrapper::array_interface)
        .def_property_readonly("__cuda_array_interface__", &StorageViewWrapper::cuda_array_interface)

        .def("__str__", &StorageViewWrapper::str)
        ;
    }

  }
}
