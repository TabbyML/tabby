import os
import pybind11

from setuptools import setup, find_packages, Extension


ctranslate2_root = os.getenv("CTRANSLATE2_ROOT", "/usr/local")
ctranslate2_module = Extension(
    "ctranslate2.translator",
    sources=["translator.cc"],
    extra_compile_args=["-std=c++11"],
    include_dirs=["%s/include" % ctranslate2_root, pybind11.get_include()],
    library_dirs=["%s/lib" % ctranslate2_root],
    libraries=["ctranslate2"])

setup(
    name="ctranslate2",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[ctranslate2_module],
    install_requires=[]
)
