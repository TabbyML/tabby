import os
import sys

import pytest


def get_data_dir():
    data_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "..", "tests", "data"
    )

    # Verify that downloaded files are present.
    translit_model = os.path.join(data_dir, "models", "transliteration-aren-all")
    if not os.path.isdir(translit_model):
        pytest.skip("Data files are not available")

    return data_dir


def write_tokens(batch_tokens, path):
    with open(path, "w", encoding="utf-8") as f:
        for tokens in batch_tokens:
            f.write(" ".join(tokens))
            f.write("\n")


skip_on_windows = pytest.mark.skipif(
    sys.platform == "win32", reason="Test case disabled on Windows"
)

only_on_linux = pytest.mark.skipif(
    sys.platform != "linux", reason="Test case only running on Linux"
)
