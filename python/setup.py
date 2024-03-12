import glob
import os
import sys

import pybind11

from pybind11.setup_helpers import ParallelCompile
from setuptools import Extension, find_packages, setup

base_dir = os.path.dirname(os.path.abspath(__file__))
include_dirs = [pybind11.get_include()]
library_dirs = []


def _get_long_description():
    readme_path = os.path.join(base_dir, "README.md")
    if not os.path.exists(readme_path):
        return ""
    with open(readme_path, encoding="utf-8") as readme_file:
        return readme_file.read()


def _get_project_version():
    version_path = os.path.join(base_dir, "ctranslate2", "version.py")
    version = {}
    with open(version_path, encoding="utf-8") as fp:
        exec(fp.read(), version)
    return version["__version__"]


def _maybe_add_library_root(lib_name):
    if "%s_ROOT" % lib_name in os.environ:
        root = os.environ["%s_ROOT" % lib_name]
        include_dirs.append("%s/include" % root)
        for lib_dir in ("lib", "lib64"):
            path = "%s/%s" % (root, lib_dir)
            if os.path.exists(path):
                library_dirs.append(path)
                break


_maybe_add_library_root("CTRANSLATE2")

cflags = ["-std=c++17", "-fvisibility=hidden"]
ldflags = []
package_data = {}
if sys.platform == "darwin":
    # std::visit requires macOS 10.14
    cflags.append("-mmacosx-version-min=10.14")
    ldflags.append("-Wl,-rpath,/usr/local/lib")
elif sys.platform == "win32":
    cflags = ["/std:c++17", "/d2FH4-"]
    package_data["ctranslate2"] = ["*.dll"]

ctranslate2_module = Extension(
    "ctranslate2._ext",
    sources=glob.glob(os.path.join("cpp", "*.cc")),
    extra_compile_args=cflags,
    extra_link_args=ldflags,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=["ctranslate2"],
)

ParallelCompile("CMAKE_BUILD_PARALLEL_LEVEL").install()

setup(
    name="ctranslate2",
    version=_get_project_version(),
    license="MIT",
    description="Fast inference engine for Transformer models",
    long_description=_get_long_description(),
    long_description_content_type="text/markdown",
    author="OpenNMT",
    url="https://opennmt.net",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: GPU :: NVIDIA CUDA :: 12 :: 12.0",
        "Environment :: GPU :: NVIDIA CUDA :: 12 :: 12.1",
        "Environment :: GPU :: NVIDIA CUDA :: 12 :: 12.2",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        "Documentation": "https://opennmt.net/CTranslate2",
        "Forum": "https://forum.opennmt.net",
        "Gitter": "https://gitter.im/OpenNMT/CTranslate2",
        "Source": "https://github.com/OpenNMT/CTranslate2",
    },
    keywords="opennmt nmt neural machine translation cuda mkl inference quantization",
    packages=find_packages(exclude=["bin"]),
    package_data=package_data,
    ext_modules=[ctranslate2_module],
    python_requires=">=3.8",
    install_requires=[
        "setuptools",
        "numpy",
        "pyyaml>=5.3,<7",
    ],
    entry_points={
        "console_scripts": [
            "ct2-fairseq-converter=ctranslate2.converters.fairseq:main",
            "ct2-marian-converter=ctranslate2.converters.marian:main",
            "ct2-openai-gpt2-converter=ctranslate2.converters.openai_gpt2:main",
            "ct2-opennmt-py-converter=ctranslate2.converters.opennmt_py:main",
            "ct2-opennmt-tf-converter=ctranslate2.converters.opennmt_tf:main",
            "ct2-opus-mt-converter=ctranslate2.converters.opus_mt:main",
            "ct2-transformers-converter=ctranslate2.converters.transformers:main",
        ],
    },
)
