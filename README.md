# CTranslate2

CTranslate2 is a fast C++ inference engine for OpenNMT models. It currently focuses on CPU translation of Transformer models trained with OpenNMT-tf.

## Requirements

* C++11
* CMake (>= 3.7)
* Intel® MKL (>= 2018)

When compiling with `-DLIB_ONLY=OFF` (default) or the Python bindings:

* Boost

When compiling with `-DWITH_CUDA=ON`:

* CUDA (>= 8.0)
* cuDNN (>= 7.1)
* TensorRT (>= 4.0)

## Compiling

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```

## Building Docker images

The Docker files require the following dependencies in the `deps/` directory:

1. [Intel® MKL](https://software.intel.com/en-us/mkl/choose-download/linux)
2. [TensorRT](https://developer.nvidia.com/tensorrt)

```bash
docker build -t systran/ctranslate2:v0.5.0 -f Dockerfile .
docker build -t systran/ctranslate2_gpu:v0.5.0 -f Dockerfile.cuda .
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
