# Installation

## Install with pip

Binary Python wheels are published on [PyPI](https://pypi.org/project/ctranslate2/) and can be directly installed with pip:

```bash
pip install ctranslate2
```

The Python wheels have the following requirements:

* OS: Linux (x86-64, AArch64), macOS (x86-64, ARM64), Windows (x86-64)
* Python version: >= 3.7
* pip version: >= 19.3 to support `manylinux2014` wheels

```{admonition} GPU support
The Linux and Windows Python wheels support GPU execution. Install [CUDA](https://developer.nvidia.com/cuda-toolkit) 11.2 or above to use the GPU.

If you plan to run models with convolutional layers (e.g. for speech recognition), you should also install [cuDNN 8](https://developer.nvidia.com/cudnn).
```

## Install with Docker

Docker images can be downloaded from the [GitHub Container registry](https://github.com/OpenNMT/CTranslate2/pkgs/container/ctranslate2):

```bash
docker pull ghcr.io/opennmt/ctranslate2:latest-ubuntu20.04-cuda11.2
```

The images include:

* the NVIDIA libraries cuBLAS and cuDNN to support GPU execution
* the C++ library installed in `/opt/ctranslate2`
* the Python module installed in the Python system packages
* the translator executable, which is the image entrypoint:

```bash
docker run --rm ghcr.io/opennmt/ctranslate2:latest-ubuntu20.04-cuda11.2 --help
```

```{admonition} GPU support
The Docker image supports GPU execution. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html) to use GPUs from Docker.
```

## Install from sources

### Download the source code

Clone the CTranslate2 Git repository and its submodules.

```bash
git clone --recursive https://github.com/OpenNMT/CTranslate2.git
```

### Compile the C++ library

Compiling the library requires a compiler supporting C++17 and CMake 3.15 or greater.

```bash
mkdir build && cd build
cmake ..
make -j4
make install
```

By default, the library is compiled with the [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) backend which should be installed separately. See the {ref}`installation:build options` to select or add another backend.

### Compile the Python wrapper

Once the C++ library is installed, you can compile the Python wrapper which uses [pybind11](https://github.com/pybind/pybind11). This step requires the Python development libraries to be installed on the system.

```bash
cd python
pip install -r install_requirements.txt
python setup.py bdist_wheel
pip install dist/*.whl
```

```{attention}
If you installed the C++ library in a custom directory, you should configure additional environment variables:

* When running `setup.py`, set `CTRANSLATE2_ROOT` to the CTranslate2 install directory.
* When running your Python application, add the CTranslate2 library path to `LD_LIBRARY_PATH`.
```

### Build options

The following options can be set with `-DOPTION=VALUE` during the CMake configuration:

| CMake option | Values (default in bold) | Description |
| --- | --- | --- |
| BUILD_CLI | OFF, **ON** | Compiles the command line clients |
| BUILD_TESTS | **OFF**, ON | Compiles the tests |
| CMAKE_CXX_FLAGS | *compiler flags* | Defines additional compiler flags |
| CMAKE_INSTALL_PREFIX | *path* | Defines the installation path of the library |
| CUDA_ARCH_LIST | **Auto** | List of CUDA architectures to compile for (see [`cuda_select_nvcc_arch_flags`](https://cmake.org/cmake/help/latest/module/FindCUDA.html) in the CMake documentation) |
| CUDA_DYNAMIC_LOADING | **OFF**, ON | Enables the dynamic loading of CUDA libraries at runtime instead of linking against them (requires CUDA >= 11) |
| CUDA_NVCC_FLAGS | *compiler flags* | Defines additional compilation flags for `nvcc` |
| ENABLE_CPU_DISPATCH | OFF, **ON** | Compiles CPU kernels for multiple ISA and dispatches at runtime (should be disabled when explicitly targeting an architecture with the `-march` compilation flag) |
| ENABLE_PROFILING | **OFF**, ON | Enables the integrated profiler (usually disabled in production builds) |
| OPENMP_RUNTIME | **INTEL**, COMP, NONE | Selects or disables the OpenMP runtime:<ul><li>INTEL: Intel OpenMP</li><li>COMP: OpenMP runtime provided by the compiler</li><li>NONE: no OpenMP runtime</li></ul> |
| WITH_CUDA | **OFF**, ON | Compiles with the CUDA backend |
| WITH_CUDNN | **OFF**, ON | Compiles with the cuDNN backend |
| WITH_DNNL | **OFF**, ON | Compiles with the oneDNN backend (a.k.a. DNNL) |
| WITH_MKL | OFF, **ON** | Compiles with the Intel MKL backend |
| WITH_ACCELERATE | **OFF**, ON | Compiles with the Apple Accelerate backend |
| WITH_OPENBLAS | **OFF**, ON | Compiles with the OpenBLAS backend |
| WITH_RUY | **OFF**, ON | Compiles with the Ruy backend |

Some build options require additional dependencies. See their respective documentation for installation instructions.

* `-DWITH_CUDA=ON` requires [CUDA](https://developer.nvidia.com/cuda-toolkit) >= 11.0
* `-DWITH_CUDNN=ON` requires [cuDNN](https://developer.nvidia.com/cudnn) >= 8
* `-DWITH_MKL=ON` requires [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) >= 2019.5
* `-DWITH_DNNL=ON` requires [oneDNN](https://github.com/oneapi-src/oneDNN) >= 3.0
* `-DWITH_ACCELERATE=ON` requires [Accelerate](https://developer.apple.com/documentation/accelerate)
* `-DWITH_OPENBLAS=ON` requires [OpenBLAS](https://github.com/xianyi/OpenBLAS)

Multiple backends can be enabled for a single build, for example:

* `-DWITH_MKL=ON -DWITH_CUDA=ON`: enable CPU and GPU support
* `-DWITH_MKL=ON -DWITH_DNNL=ON`: during runtime, the library will select Intel MKL when running on Intel and oneDNN when running on AMD
* `-DWITH_OPENBLAS=ON -DWITH_RUY=ON`: use Ruy for quantized models and OpenBLAS for non quantized models
