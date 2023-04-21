#pragma once

#include <chrono>
#include <future>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ctranslate2/types.h>

namespace py = pybind11;

namespace ctranslate2 {
  namespace python {

    using StringOrMap = std::variant<std::string, std::unordered_map<std::string, std::string>>;
    using Tokens = std::vector<std::string>;
    using Ids = std::vector<size_t>;
    using BatchTokens = std::vector<Tokens>;
    using BatchIds = std::vector<Ids>;
    using EndToken = std::variant<std::string, std::vector<std::string>, std::vector<size_t>>;

    class ComputeTypeResolver {
    private:
      const std::string _device;

    public:
      ComputeTypeResolver(std::string device)
        : _device(std::move(device)) {
      }

      ComputeType
      operator()(const std::string& compute_type) const {
        return str_to_compute_type(compute_type);
      }

      ComputeType
      operator()(const std::unordered_map<std::string, std::string>& compute_type) const {
        auto it = compute_type.find(_device);
        if (it == compute_type.end())
          return ComputeType::DEFAULT;
        return operator()(it->second);
      }
    };

    class DeviceIndexResolver {
    public:
      std::vector<int> operator()(int device_index) const {
        return {device_index};
      }

      std::vector<int> operator()(const std::vector<int>& device_index) const {
        return device_index;
      }
    };

    template <typename T>
    class AsyncResult {
    public:
      AsyncResult(std::future<T> future)
        : _future(std::move(future))
      {
      }

      const T& result() {
        if (!_done) {
          {
            py::gil_scoped_release release;
            try {
              _result = _future.get();
            } catch (...) {
              _exception = std::current_exception();
            }
          }
          _done = true;  // Assign done attribute while the GIL is held.
        }
        if (_exception)
          std::rethrow_exception(_exception);
        return _result;
      }

      bool done() {
        constexpr std::chrono::seconds zero_sec(0);
        return _done || _future.wait_for(zero_sec) == std::future_status::ready;
      }

    private:
      std::future<T> _future;
      T _result;
      bool _done = false;
      std::exception_ptr _exception;
    };

    template <typename Result>
    std::vector<Result> wait_on_futures(std::vector<std::future<Result>> futures) {
      std::vector<Result> results;
      results.reserve(futures.size());
      for (auto& future : futures)
        results.emplace_back(future.get());
      return results;
    }

    template <typename Result>
    std::variant<std::vector<Result>, std::vector<AsyncResult<Result>>>
    maybe_wait_on_futures(std::vector<std::future<Result>> futures, bool asynchronous) {
      if (asynchronous) {
        std::vector<AsyncResult<Result>> results;
        results.reserve(futures.size());
        for (auto& future : futures)
          results.emplace_back(std::move(future));
        return std::move(results);
      } else {
        return wait_on_futures(std::move(futures));
      }
    }

    template <typename T>
    static void declare_async_wrapper(py::module& m, const char* name) {
      py::class_<AsyncResult<T>>(m, name, "Asynchronous wrapper around a result object.")
        .def("result", &AsyncResult<T>::result,
             R"pbdoc(
                 Blocks until the result is available and returns it.

                 If an exception was raised when computing the result,
                 this method raises the exception.
             )pbdoc")
        .def("done", &AsyncResult<T>::done, "Returns ``True`` if the result is available.")
        ;
    }

  }
}
