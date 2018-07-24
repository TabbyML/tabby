# CTranslate2

CTranslate2 is a fast C++ inference engine for OpenNMT models. It currently focuses on CPU translation of Transformer models trained with OpenNMT-tf.

## Requirements

* C++11
* CMake
* Intel® MKL 2018
* Boost

## Compiling

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```

## Using

To get started in using CTranslate2, you can download a pretrained English-German model:

```bash
scp -r devling@minigpu:/home/klein/dev/ctransformer/benchmark/data/ .
cli/translate --model data/ende_transformer_quantized --src data/valid.en.500
```

Run `cli/translate -h` to list available options.

## Testing

[Google Test](https://github.com/google/googletest) is used for testing. To run the test suite, run this command in the build directory:

```bash
./tests/ctranslate2_test
```
