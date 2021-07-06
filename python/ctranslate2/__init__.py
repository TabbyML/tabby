try:
    from ctranslate2.translator import Translator
    from ctranslate2.translator import contains_model
    from ctranslate2.translator import get_cuda_device_count
    from ctranslate2.translator import get_supported_compute_types
except ImportError as e:
    # Allow using the Python package without the compiled translator extension.
    if "No module named" in str(e):
        pass
    else:
        raise

from ctranslate2 import converters
from ctranslate2 import specs

from ctranslate2.version import __version__
