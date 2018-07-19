#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>

#include <ctranslate2/translator_pool.h>
#include <ctranslate2/utils.h>

namespace py = boost::python;

class GILReleaser {
public:
  GILReleaser()
    : _save_state(PyEval_SaveThread()) {
  }
  ~GILReleaser() {
    PyEval_RestoreThread(_save_state);
  }
private:
  PyThreadState* _save_state;
};

class TranslatorWrapper
{
public:
  TranslatorWrapper(const std::string& model_path,
                    const std::string& model_type,
                    size_t max_decoding_steps,
                    size_t beam_size,
                    float length_penalty,
                    size_t inter_threads,
                    size_t intra_threads)
    : _translator_pool(inter_threads,
                       ctranslate2::ModelFactory::load(model_type, model_path),
                       max_decoding_steps,
                       beam_size,
                       length_penalty,
                       "") {
    ctranslate2::init(intra_threads);
  }

  py::list translate_batch(const py::object& tokens) {
    if (tokens == py::object())
      return py::list();

    std::vector<std::vector<std::string>> tokens_vec;

    for (auto it = py::stl_input_iterator<py::list>(tokens);
         it != py::stl_input_iterator<py::list>(); it++) {
      tokens_vec.emplace_back(py::stl_input_iterator<std::string>(*it),
                              py::stl_input_iterator<std::string>());
    }

    auto result_vec = translate(tokens_vec);

    py::list result;
    for (size_t i = 0; i < result_vec.size(); i++) {
      py::list temp;
      for (size_t t = 0; t < result_vec[i].size(); t++) {
        temp.append(result_vec[i][t]);
      }
      result.append(temp);
    }

    return result;
  }

private:
  std::vector<std::vector<std::string>>
  translate(const std::vector<std::vector<std::string>>& input) {
    GILReleaser releaser;
    auto future = _translator_pool.post(input);
    future.wait();
    return future.get();
  }

  ctranslate2::TranslatorPool _translator_pool;
};

BOOST_PYTHON_MODULE(translator)
{
  PyEval_InitThreads();
  py::class_<TranslatorWrapper, boost::noncopyable>(
      "Translator",
      py::init<std::string, std::string, size_t, size_t, float, size_t, size_t>(
        (py::arg("max_decoding_steps")=250,
         py::arg("beam_size")=4,
         py::arg("length_penalty")=0.6,
         py::arg("inter_threads")=1,
         py::arg("intra_threads")=4)))
    .def("translate_batch", &TranslatorWrapper::translate_batch)
    ;
}
