import sys

import numpy as np
import pytest
import test_utils

import ctranslate2


def _assert_same_array(a, b):
    assert a["shape"] == b["shape"]
    assert a["data"] == b["data"]
    assert a["typestr"] == b["typestr"]


@pytest.mark.parametrize(
    "dtype,name",
    [
        (np.int8, "int8"),
        (np.int16, "int16"),
        (np.int32, "int32"),
        (np.float16, "float16"),
        (np.float32, "float32"),
    ],
)
def test_storageview_cpu(dtype, name):
    x = np.ones((2, 4), dtype=dtype)
    s = ctranslate2.StorageView.from_array(x)

    assert s.device == "cpu"
    assert s.device_index == 0
    assert s.dtype.name == name
    assert s.shape == [2, 4]

    _assert_same_array(s.__array_interface__, x.__array_interface__)

    with pytest.raises(AttributeError, match="CPU"):
        s.__cuda_array_interface__

    assert str(s) == " 1 1 1 ... 1 1 1\n[cpu:0 %s storage viewed as 2x4]" % name

    x[0][2] = 3
    x[1][3] = 8
    assert str(s) == " 1 1 3 ... 1 1 8\n[cpu:0 %s storage viewed as 2x4]" % name

    y = np.array(x)
    assert test_utils.array_equal(x, y)


def test_storageview_to_device():
    x = np.ones(10, dtype="float32")
    cx = ctranslate2.StorageView.from_array(x)

    assert cx.device == "cpu"
    assert cx.dtype == ctranslate2.DataType.float32

    cpu_x = cx.to_device(ctranslate2.Device.cpu)
    assert test_utils.array_equal(x, np.array(cpu_x))


@test_utils.require_cuda
def test_storageview_cuda():
    import torch

    x = torch.ones((2, 4), device="cuda")
    s = ctranslate2.StorageView.from_array(x)

    assert s.device == "cuda"
    assert s.device_index == 0
    assert s.dtype == ctranslate2.DataType.float32
    assert s.shape == [2, 4]

    _assert_same_array(s.__cuda_array_interface__, x.__cuda_array_interface__)

    with pytest.raises(AttributeError, match="CUDA"):
        s.__array_interface__

    assert str(s) == " 1 1 1 ... 1 1 1\n[cuda:0 float32 storage viewed as 2x4]"

    x[0][2] = 3
    x[1][3] = 8
    assert str(s) == " 1 1 3 ... 1 1 8\n[cuda:0 float32 storage viewed as 2x4]"

    y = torch.as_tensor(s, device="cuda")
    _assert_same_array(s.__cuda_array_interface__, y.__cuda_array_interface__)


@test_utils.require_cuda
def test_storageview_cuda_to_device():
    x = np.ones(10, dtype="float32")
    # convert to cuda tensor
    cuda_x = ctranslate2.StorageView.from_array(x).to_device(ctranslate2.Device.cuda)
    assert cuda_x.device == "cuda"

    # modify original tensor and convert back
    x *= 2
    cpu_x = np.array(cuda_x.to_device(ctranslate2.Device.cpu))

    assert cpu_x.dtype == x.dtype
    assert x.sum() == 2 * cpu_x.sum()


def test_storageview_conversion():
    x = np.ones((2, 4), dtype=np.float32)
    s = ctranslate2.StorageView.from_array(x)
    assert s.dtype == ctranslate2.DataType.float32

    s = s.to(ctranslate2.DataType.bfloat16)
    assert s.dtype == ctranslate2.DataType.bfloat16

    with pytest.raises(RuntimeError, match="not supported in the array interface"):
        s.__array_interface__


def test_storageview_strides():
    x = np.ones((2, 4), dtype=np.float32)
    x_t = x.transpose()
    with pytest.raises(ValueError, match="contiguous"):
        ctranslate2.StorageView.from_array(x_t)


def test_storageview_readonly():
    x = np.ones((2, 4), dtype=np.float32)
    x.flags.writeable = False
    with pytest.raises(ValueError, match="read-only"):
        ctranslate2.StorageView.from_array(x)


def test_storageview_reference():
    x = np.ones((2, 4), dtype=np.float32)
    refcount_before = sys.getrefcount(x)
    s = ctranslate2.StorageView.from_array(x)
    refcount_after = sys.getrefcount(x)
    assert refcount_after == refcount_before + 1
    del s
    refcount_after_del = sys.getrefcount(x)
    assert refcount_after_del == refcount_before
