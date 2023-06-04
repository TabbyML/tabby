import shutil

import pytest


@pytest.fixture
def tmp_dir(tmpdir_factory):
    path = tmpdir_factory.mktemp("workspace")
    yield path
    shutil.rmtree(path)
