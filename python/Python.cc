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
                    const std::string& vocab_mapping,
                    size_t thread_pool_size)
    : _translator_pool(thread_pool_size,
                       ctranslate2::ModelFactory::load(model_type, model_path),
                       vocab_mapping) {
  }

  py::list translate_batch(const py::object& tokens,
                           size_t beam_size,
                           size_t num_hypotheses,
                           float length_penalty,
                           size_t max_decoding_steps) {
    if (tokens == py::object())
      return py::list();

    std::vector<std::vector<std::string>> tokens_vec;

    for (auto it = py::stl_input_iterator<py::list>(tokens);
         it != py::stl_input_iterator<py::list>(); it++) {
      tokens_vec.emplace_back(py::stl_input_iterator<std::string>(*it),
                              py::stl_input_iterator<std::string>());
    }

    auto options = ctranslate2::TranslationOptions();
    options.beam_size = beam_size;
    options.length_penalty = length_penalty;
    options.max_decoding_steps = max_decoding_steps;
    options.num_hypotheses = num_hypotheses;

    std::vector<ctranslate2::TranslationResult> results;

    {
      GILReleaser releaser;
      results = std::move(_translator_pool.post(tokens_vec, options).get());
    }

    py::list py_results;
    for (auto& result : results) {
      py::list temp;
      for (const auto& token : result.output())
        temp.append(token);
      py_results.append(temp);
    }

    return py_results;
  }

private:
  ctranslate2::TranslatorPool _translator_pool;
};

BOOST_PYTHON_MODULE(translator)
{
  PyEval_InitThreads();
  py::def("initialize", &ctranslate2::init);
  py::class_<TranslatorWrapper, boost::noncopyable>(
    "Translator",
    py::init<std::string, std::string, std::string, size_t>(
      (py::arg("vocab_mapping")="",
       py::arg("thread_pool_size")=1)))
    .def("translate_batch", &TranslatorWrapper::translate_batch,
         (py::arg("beam_size")=4,
          py::arg("num_hypotheses")=1,
          py::arg("length_penalty")=0.6,
          py::arg("max_decoding_steps")=250))
    ;
}
