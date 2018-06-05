#pragma once

#include <string>
#include <vector>

#include "vocabulary.h"
#include "encoder.h"
#include "decoder.h"

namespace opennmt {

  void translate(const std::vector<std::vector<std::string> >& input_tokens,
                 const Vocabulary& vocabulary,
                 Encoder& encoder,
                 Decoder& decoder,
                 size_t beam_size);

}
