#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>

#include <opennmt/translator.h>
#include <opennmt/utils.h>

namespace py = boost::python;

void set_intra_threads(int n) {
  opennmt::set_num_threads(n);
}

class TranslatorWrapper
{
public:
  TranslatorWrapper(const std::string& model_path,
                    const std::string& model_type,
                    size_t max_decoding_steps,
                    size_t beam_size,
                    float length_penalty)
    : _translator(opennmt::ModelFactory::load(model_type, model_path),
                  max_decoding_steps,
                  beam_size,
                  length_penalty,
                  "") {
    opennmt::init(4);
  }

  TranslatorWrapper shallow_copy() {
    return TranslatorWrapper(*this);
  }

  py::list translate_batch(const py::object& tokens) {
    if (tokens == py::object())
      return py::list();

    std::vector<std::vector<std::string>> tokens_vec;

    for (auto it = py::stl_input_iterator<py::list>(tokens);
         it != py::stl_input_iterator<py::list>(); it++) {
      tokens_vec.push_back(std::vector<std::string>(py::stl_input_iterator<std::string>(*it),
                                                    py::stl_input_iterator<std::string>()));
    }

    auto result_vec = _translator.translate_batch(tokens_vec);

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
  opennmt::Translator _translator;
};

BOOST_PYTHON_MODULE(translator)
{
  py::def("set_intra_threads", &set_intra_threads);
  py::class_<TranslatorWrapper>(
      "Translator",
      py::init<std::string, std::string, size_t, size_t, float>(
        (py::arg("max_decoding_steps")=250,
         py::arg("beam_size")=4,
         py::arg("length_penalty")=0.6)))
    .def("translate_batch", &TranslatorWrapper::translate_batch)
    .def("__copy__", &TranslatorWrapper::shallow_copy)
    ;
}
