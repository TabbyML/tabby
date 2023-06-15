#pragma once

#include <istream>
#include <memory>
#include <string>
#include <unordered_map>

#include "ctranslate2/vocabulary.h"

namespace ctranslate2 {
  namespace models {

    // The ModelReader interface allows user code to customize how and where to read model files.
    class ModelReader {
    public:
      virtual ~ModelReader() = default;

      // Returns a string identifying the model to be loaded (e.g. its path on disk).
      virtual std::string get_model_id() const = 0;
      // Returns a stream over a file included in the model, or nullptr if the file can't be openned.
      virtual std::unique_ptr<std::istream> get_file(const std::string& filename,
                                                     const bool binary = false) = 0;

      // Wrapper around get_file, raises an exception if the file can't be openned.
      std::unique_ptr<std::istream> get_required_file(const std::string& filename,
                                                      const bool binary = false);
    };

    class ModelFileReader : public ModelReader {
    public:
      ModelFileReader(std::string model_dir);
      std::string get_model_id() const override;
      std::unique_ptr<std::istream> get_file(const std::string& filename,
                                             const bool binary = false) override;

    private:
      std::string _model_dir;
    };

    class ModelMemoryReader : public ModelReader {
    public:
      ModelMemoryReader(std::string model_name);

      void register_file(std::string filename, std::string content);

      std::string get_model_id() const override;
      std::unique_ptr<std::istream> get_file(const std::string& filename,
                                             const bool binary = false) override;

    private:
      std::string _model_name;
      std::unordered_map<std::string, std::string> _files;
    };


    std::shared_ptr<Vocabulary>
    load_vocabulary(ModelReader& model_reader,
                    const std::string& filename,
                    VocabularyInfo vocab_info);

  }
}
