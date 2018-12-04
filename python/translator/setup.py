import os

from setuptools import setup, Extension


include_dirs = []
library_dirs = []

def _maybe_add_library_root(lib_name):
  if "%s_ROOT" % lib_name in os.environ:
    root = os.environ["%s_ROOT" % lib_name]
    include_dirs.append("%s/include" % root)
    library_dirs.append("%s/lib" % root)

_maybe_add_library_root("BOOST")
_maybe_add_library_root("CTRANSLATE")

ctranslate2_module = Extension(
    "ctranslate2.translator",
    sources=["Python.cc"],
    extra_compile_args=["-std=c++11"],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=[os.getenv("BOOST_PYTHON_LIBRARY", "boost_python"), "ctranslate2"])

setup(
    name="ctranslate2",
    version="0.1.0",
    packages=["ctranslate2"],
    ext_modules=[ctranslate2_module],
    install_requires=[]
)
