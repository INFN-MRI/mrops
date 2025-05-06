"""Test backend utilities"""

import pytest

import numpy as np
import torch

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from pygrog import backend


def test_device_cpu_creation():
    d = backend.Device(-1)
    assert d.id == -1
    assert d.xp is np
    assert d.torch is torch
    assert int(d) == -1
    assert d == -1
    assert repr(d) == "<CPU Device>"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_gpu_creation():
    d = backend.Device(0)
    assert d.id == 0
    assert d.xp.__name__ == "cupy"
    assert repr(d) == "<CUDA Device 0>"


def test_device_from_str():
    d1 = backend.Device("cpu")
    assert d1.id == -1
    if torch.cuda.is_available():
        d2 = backend.Device("cuda:0")
        assert d2.id == 0


def test_device_comparison():
    cpu = backend.Device(-1)
    assert cpu == backend.cpu_device
    if torch.cuda.is_available():
        gpu = backend.Device("cuda:0")
        assert gpu == backend.Device(0)
        assert gpu != cpu


def test_get_array_module_numpy():
    arr = np.array([1, 2, 3])
    mod = backend.get_array_module(arr)
    assert mod is np


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_get_array_module_torch():
    arr = torch.tensor([1, 2, 3])
    mod = backend.get_array_module(arr)
    assert mod is torch


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
def test_get_array_module_cupy():
    arr = cp.array([1, 2, 3])
    mod = backend.get_array_module(arr)
    assert mod is cp


def test_get_device_numpy():
    arr = np.array([1])
    d = backend.get_device(arr)
    assert isinstance(d, backend.Device)
    assert d == -1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_get_device_torch():
    arr = torch.tensor([1]).cuda()
    d = backend.get_device(arr)
    assert isinstance(d, backend.Device)
    assert d.id == 0


def test_to_device_same_device():
    arr = np.array([1, 2, 3])
    out = backend.to_device(arr, -1)
    assert np.array_equal(arr, out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_to_device_to_gpu_and_back():
    arr = np.array([1, 2, 3])
    gpu_device = backend.Device(0)
    gpu_arr = backend.to_device(arr, gpu_device)
    assert hasattr(gpu_arr, "device")

    back_to_cpu = backend.to_device(gpu_arr, -1)
    assert np.array_equal(back_to_cpu, arr)


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
def test_to_array_module_cupy():
    arr = np.array([1, 2])
    converted = backend.to_array_module(arr, "cupy")
    assert isinstance(converted, cp.ndarray)


def test_to_array_module_numpy():
    arr = torch.tensor([1.0, 2.0])
    converted = backend.to_array_module(arr, "numpy")
    assert isinstance(converted, np.ndarray)


def test_to_array_module_torch():
    arr = np.array([1.0, 2.0])
    converted = backend.to_array_module(arr, "torch")
    assert isinstance(converted, torch.Tensor)
