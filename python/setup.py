import os
import sys
import pybind11

from setuptools import setup, find_packages, Extension


include_dirs = [pybind11.get_include()]
library_dirs = []


def _get_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(readme_path, encoding="utf-8") as readme_file:
        return readme_file.read()


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

cflags = ["-std=c++17"]
if sys.platform == "darwin":
    # std::visit requires macOS 10.14
    cflags.append("-mmacosx-version-min=10.14")

ldflags = []
if sys.platform == "darwin":
    ldflags.append("-Wl,-rpath,/usr/local/lib")

ctranslate2_module = Extension(
    "ctranslate2.translator",
    sources=["translator.cc"],
    extra_compile_args=cflags,
    extra_link_args=ldflags,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=["ctranslate2"],
)

setup(
    name="ctranslate2",
    version="1.19.0",
    license="MIT",
    description="Fast inference engine for OpenNMT models",
    long_description=_get_long_description(),
    long_description_content_type="text/markdown",
    author="OpenNMT",
    author_email="guillaume.klein@systrangroup.com",
    url="https://opennmt.net",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: GPU :: NVIDIA CUDA :: 10.1",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        "Forum": "https://forum.opennmt.net",
        "Gitter": "https://gitter.im/OpenNMT/CTranslate2",
        "Source": "https://github.com/OpenNMT/CTranslate2",
    },
    keywords="opennmt nmt neural machine translation cuda mkl inference quantization",
    packages=find_packages(exclude=["bin"]),
    ext_modules=[ctranslate2_module],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
    ],
    entry_points={
        "console_scripts": [
            "ct2-opennmt-py-converter=ctranslate2.bin.opennmt_py_converter:main",
            "ct2-opennmt-tf-converter=ctranslate2.bin.opennmt_tf_converter:main",
        ],
    },
)
