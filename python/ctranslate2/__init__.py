import sys

if sys.platform == "win32":
    import ctypes
    import os

    import pkg_resources

    module_name = sys.modules[__name__].__name__
    package_dir = pkg_resources.resource_filename(module_name, "")

    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        add_dll_directory(package_dir)

    for library in ("libiomp5md", "ctranslate2"):
        ctypes.CDLL(os.path.join(package_dir, "%s.dll" % library))

try:
    from ctranslate2.translator import (
        AsyncGenerationResult,
        AsyncTranslationResult,
        GenerationResult,
        Generator,
        TranslationResult,
        TranslationStats,
        Translator,
        contains_model,
        get_cuda_device_count,
        get_supported_compute_types,
        set_random_seed,
    )
except ImportError as e:
    # Allow using the Python package without the compiled translator extension.
    if "No module named" in str(e):
        pass
    else:
        raise

from ctranslate2 import converters, specs
from ctranslate2.version import __version__
