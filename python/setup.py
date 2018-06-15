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

ctranslate_module = Extension(
    "ctranslate.translator",
    sources=["Python.cc"],
    extra_compile_args=["-std=c++11"],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=[os.getenv("BOOST_PYTHON_LIBRARY", "boost_python"), "opennmt"])

setup(
    name="ctranslate",
    version="0.1.0",
    packages=["ctranslate"],
    ext_modules=[ctranslate_module]
)
