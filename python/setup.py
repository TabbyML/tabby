import os
import pybind11

from setuptools import setup, find_packages, Extension


include_dirs = [pybind11.get_include()]
library_dirs = []

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

ctranslate2_module = Extension(
    "ctranslate2.translator",
    sources=["translator.cc"],
    extra_compile_args=["-std=c++11"],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=["ctranslate2"])

setup(
    name="ctranslate2",
    version="1.3.0",
    license="MIT",
    description="Custom C++ inference engine for OpenNMT models",
    author="OpenNMT",
    author_email="guillaume.klein@systrangroup.com",
    url="https://github.com/OpenNMT/CTranslate2",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    project_urls={
        "Forum": "http://forum.opennmt.net/",
        "Gitter": "https://gitter.im/OpenNMT/CTranslate2",
        "Source": "https://github.com/OpenNMT/CTranslate2/"
    },
    keywords="opennmt nmt neural machine translation cuda mkl inference quantization",
    packages=find_packages(exclude=["bin"]),
    ext_modules=[ctranslate2_module],
    install_requires=[
        "numpy",
        "six",
    ],
    entry_points={
        "console_scripts": [
            "ct2-opennmt-py-converter=ctranslate2.bin.opennmt_py_converter:main",
            "ct2-opennmt-tf-converter=ctranslate2.bin.opennmt_tf_converter:main",
        ],
    }
)
