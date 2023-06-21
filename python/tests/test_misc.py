import pytest

from ctranslate2.extensions import _batch_iterator as batch_iterator


@pytest.mark.parametrize(
    "batch_size,batch_type,lengths,expected_batch_sizes",
    [
        (2, "examples", [2, 3, 4, 1, 1], [2, 2, 1]),
        (6, "tokens", [2, 3, 1, 4, 1, 2], [2, 1, 1, 2]),
    ],
)
def test_batch_iterator(batch_size, batch_type, lengths, expected_batch_sizes):
    iterable = (["a"] * length for length in lengths)

    batches = batch_iterator(iterable, batch_size, batch_type)
    batch_sizes = [len(batch[0]) for batch in batches]

    assert batch_sizes == expected_batch_sizes
