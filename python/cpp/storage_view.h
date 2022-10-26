#pragma once

#include <pybind11/pybind11.h>

#include <ctranslate2/storage_view.h>

namespace py = pybind11;

namespace ctranslate2 {
  namespace python {

    class StorageViewWrapper {
    public:
      StorageViewWrapper(StorageView view);

      static StorageViewWrapper from_array(py::object array);

      StorageView& get_view() {
        return _view;
      }

      const StorageView& get_view() const {
        return _view;
      }

      void set_data_owner(py::object array) {
        _data_owner = array;
      }

      py::dict array_interface() const;
      py::dict cuda_array_interface() const;

      std::string str() const;

    private:
      py::object _data_owner;
      StorageView _view;
    };

  }
}
