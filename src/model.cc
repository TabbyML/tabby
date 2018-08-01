#include "ctranslate2/model.h"

#include "ctranslate2/transformer.h"
#include "ctranslate2/utils.h"

namespace ctranslate2 {

  Model::Model(const std::string& path, Device device)
    : _device(device)
    , _source_vocabulary(path + "/source_vocabulary.txt")
    , _target_vocabulary(path + "/target_vocabulary.txt")
    , _vocabulary_map(path + "/vmap.txt", _target_vocabulary) {
  }

  Device Model::device() const {
    return _device;
  }

  const Vocabulary& Model::get_source_vocabulary() const {
    return _source_vocabulary;
  }

  const Vocabulary& Model::get_target_vocabulary() const {
    return _target_vocabulary;
  }

  const VocabularyMap& Model::get_vocabulary_map() const {
    return _vocabulary_map;
  }

  StorageView Model::load_data(const Shape& shape, size_t data_width, void* data) const {
    if (data_width == 4) {
      return StorageView(shape, reinterpret_cast<float*>(data)).to(_device);
    } else if (data_width == 2) {
      StorageView s_data(shape, reinterpret_cast<int16_t*>(data));
#ifdef WITH_MKL
      if (_device == Device::CPU && support_avx2())
        return s_data.to(Device::CPU);
      else
#endif
      {
        // int16 GEMM is not optimized prior AVX2 so fallback to float.
        static const ops::Unquantize unquantize_op(1000);
        StorageView s_data_cast;
        unquantize_op(s_data, s_data_cast);
        return s_data_cast.to(_device);
      }
    }

    throw std::runtime_error("unsupported data type");
  }

  std::shared_ptr<Model> ModelFactory::load(const std::string& type,
                                            const std::string& path,
                                            Device device) {
    if (type == "transformer")
      return load(ModelType::Transformer, path, device);
    return nullptr;
  }

  std::shared_ptr<Model> ModelFactory::load(ModelType type,
                                            const std::string& path,
                                            Device device) {
    Model* model = nullptr;

    switch (type) {
    case ModelType::Transformer:
      model = new TransformerModel(path, device);
      break;
    }

    return std::shared_ptr<Model>(model);
  }

}
