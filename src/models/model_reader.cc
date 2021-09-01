#include "ctranslate2/models/model_reader.h"

#include <fstream>

namespace ctranslate2 {
  namespace models {

    std::unique_ptr<std::istream> ModelReader::get_required_file(const std::string& filename,
                                                                 const bool binary) {
      std::unique_ptr<std::istream> file = get_file(filename, binary);
      if (!file)
        throw std::runtime_error("Unable to open file '" + filename
                                 + "' in model '" + get_model_id() + "'");
      return file;
    }


    ModelFileReader::ModelFileReader(std::string model_dir)
      : _model_dir(std::move(model_dir))
    {
    }

    std::string ModelFileReader::get_model_id() const {
      return _model_dir;
    }

    std::unique_ptr<std::istream> ModelFileReader::get_file(const std::string& filename,
                                                            const bool binary) {
      const std::string path = _model_dir + "/" + filename;
      const std::ios_base::openmode mode = binary ? std::ios_base::binary : std::ios_base::in;
      auto stream = std::make_unique<std::ifstream>(path, mode);
      if (!stream || !(*stream))
        return nullptr;
      return stream;
    }

  }
}
