import os
import sys

from setuptools import setup, find_packages, Extension


include_dirs = []
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

def _get_boost_python_libname():
  libname = "boost_python"
  if sys.version_info.major == 2:
    # Boost on CentOS 7 does not come with boost_python-py27 so assume boost_python
    # is always for Python 2.
    return libname
  return "%s-py%d%d" % (libname, sys.version_info.major, sys.version_info.minor)

_maybe_add_library_root("BOOST")
_maybe_add_library_root("CTRANSLATE2")

ctranslate2_module = Extension(
    "ctranslate2.translator",
    sources=["translator.cc"],
    extra_compile_args=["-std=c++11"],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=[_get_boost_python_libname(), "ctranslate2"])

setup(
    name="ctranslate2",
    version="1.2.0",
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
