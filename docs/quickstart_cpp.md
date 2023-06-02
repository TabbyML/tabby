# C++ Quickstart

Start using the CTranslate2 library in your own C++ project.

**1\. Compile and Install CTranslate2**

```bash
mkdir build && cd build
cmake ..
make -j4 install
```

It's important that the library is getting installed into a directory that is on the CMAKE_PREFIX_PATH, otherwise you can install to a custom directory, e.g.:
```bash
export CTRANSLATE_INSTALL_PATH=$(pwd)/install
cmake .. -DCMAKE_INSTALL_PREFIX=$CTRANSLATE_INSTALL_PATH
make -j4 install
```


See [installation guide][installation.md] for more info. 

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

You will need to have your input string tokenised, 
which depends on what type of model you are using. 
Have a look at the guides to get tokens from your input string.

```cpp
#include <iostream>
#include <vector>

#include "ctranslate2/translator.h"

int main(int argc, char* argv[]) {

    std::string model_path("opus-mt-en-de");
    const ctranslate2::models::ModelLoader model_loader(model_path);
    ctranslate2::Translator translator(model_loader);

    std::vector<std::vector<std::string>> batch = {{"▁Hello", "▁World", "!", "</s>"}};
    auto translation = translator.translate_batch(batch);
    for (auto &token:  translation[0].output()) {
        std::cout << token << ' ';
    }
    std::cout << std::endl;
}
```
**5\. Compile and run the example**
```bash
cmake .
make
./main
```
If you have installed the CTranslate lib to a custom path use: `cmake -DCMAKE_PREFIX_PATH=$CTRANSLATE_INSTALL_PATH .`


This code should print the output tokens:

> ▁Hall o ▁Welt </s>

If that's the case, you successfully converted and executed a translation model with CTranslate2!
(De)tokenisation is handled outside of CTranslate2, 
so make sure to properly tokenise the input and detokenise the output.