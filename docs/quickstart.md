# Quickstart

Get started with CTranslate2 with end-to-end examples using machine translation models.

```{seealso}
The [Transformers guide](guides/transformers.md) which contains a bunch of examples for various models.
```

## Python

Start using CTranslate2 from Python by converting a pretrained model and running your first translation.

**1\. Install the Python packages**

```bash
pip install ctranslate2 OpenNMT-py==2.* sentencepiece
```

**2\. Download the English-German Transformer model trained with OpenNMT-py**

```bash
wget https://s3.amazonaws.com/opennmt-models/transformer-ende-wmt-pyOnmt.tar.gz
tar xf transformer-ende-wmt-pyOnmt.tar.gz
```

**3\. Convert the model to the CTranslate2 format**

```bash
ct2-opennmt-py-converter --model_path averaged-10-epoch.pt --output_dir ende_ctranslate2
```

**4\. Translate texts with the Python API**

```python
import ctranslate2
import sentencepiece as spm

translator = ctranslate2.Translator("ende_ctranslate2/", device="cpu")
sp = spm.SentencePieceProcessor("sentencepiece.model")

input_text = "Hello world!"
input_tokens = sp.encode(input_text, out_type=str)

results = translator.translate_batch([input_tokens])

output_tokens = results[0].hypotheses[0]
output_text = sp.decode(output_tokens)

print(output_text)
```

This code should print the sentence:

> Hallo Welt!

If that's the case, you successfully converted and executed a translation model with CTranslate2! Consider browsing the other sections for more information and examples.

## C++

Start using the CTranslate2 library in your own C++ project.

**1\. Compile and Install CTranslate2**

```bash
mkdir build && cd build
cmake ..
make -j4 install
```

It is important that the library is getting installed into a directory that is on the `CMAKE_PREFIX_PATH`, otherwise you can install to a custom directory, e.g.:
```bash
export CTRANSLATE_INSTALL_PATH=$(pwd)/install
cmake .. -DCMAKE_INSTALL_PREFIX=$CTRANSLATE_INSTALL_PATH
make -j4 install
```

See the [installation guide](installation.md) for more information.

**2\. Add CTranslate2 to your CMakeLists.txt**

```cmake
cmake_minimum_required (VERSION 2.8.11)
project (CTRANSLATE2_DEMO)

find_package(ctranslate2)

add_executable (main main.cpp)
target_link_libraries(main CTranslate2::ctranslate2)
```

**3\. Have a model ready in the CTranslate2 format**

```bash
ct2-transformers-converter --model Helsinki-NLP/opus-mt-en-de --output_dir opus-mt-en-de
```

**4\. Write the translation C++ code using the API**

You will need to have your input string tokenised, which depends on what type of model you are using. Have a look at the guides to get tokens from your input string.

```cpp
#include <iostream>
#include <vector>

#include "ctranslate2/translator.h"

int main(int argc, char* argv[]) {
  const std::string model_path("opus-mt-en-de");
  const ctranslate2::models::ModelLoader model_loader(model_path);

  ctranslate2::Translator translator(model_loader);

  const std::vector<std::vector<std::string>> batch = {{"▁Hello", "▁World", "!", "</s>"}};
  const auto translation = translator.translate_batch(batch);

  for (const auto& token : translation[0].output())
    std::cout << token << ' ';
  std::cout << std::endl;
}
```

**5\. Compile and run the example**

```bash
cmake .
make
./main
```

If you have installed the CTranslate library to a custom path use:

```bash
cmake -DCMAKE_PREFIX_PATH=$CTRANSLATE_INSTALL_PATH .
```

This code should print the output tokens:

> ▁Hall o ▁Welt !

If that's the case, you successfully converted and executed a translation model with CTranslate2!

```{important}
(De)tokenisation is handled outside of CTranslate2, so make sure to properly tokenise the input and detokenise the output.
```
