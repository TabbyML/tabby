try:
    from ctranslate2.translator import Translator
    from ctranslate2.translator import contains_model
except ImportError as e:
    # Allow using the Python package without the compiled translator extension.
    if "No module named" in str(e):
        pass
    else:
        raise

from ctranslate2 import converters
from ctranslate2 import specs
