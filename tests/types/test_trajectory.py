"""Test trajectory container."""

import numpy as np
import pytest

from mrops.types import Trajectory


@pytest.fixture
def sample_trajectory():
    """Create a sample trajectory for testing."""
    kx = np.random.rand(1, 1, 3, 2, 5)
    ky = np.random.rand(1, 1, 3, 2, 5)
    kz = np.random.rand(1, 1, 3, 1, 1)
    contrast_axis = np.random.rand(1, 3, 1, 1, 1)
    time_axis = np.random.rand(2, 1, 1, 1, 1)

    return Trajectory(
        kx=kx, ky=ky, kz=kz, contrast_axis=contrast_axis, time_axis=time_axis
    )


def test_backend_conversion_numpy(sample_trajectory):
    """Test that `to()` converts arrays to the correct numpy backend."""
    traj = sample_trajectory
    traj.to("numpy", dtype=np.float32)

    assert traj.backend == "numpy"
    assert traj.kx.dtype == np.float32
    assert traj.ky.dtype == np.float32
    assert traj.kz.dtype == np.float32
    assert traj.contrast_axis.dtype == np.float32
    assert traj.time_axis.dtype == np.float32


def test_backend_conversion_torch(sample_trajectory):
    """Test that `to()` converts arrays to the correct torch backend."""
    traj = sample_trajectory
    traj.to("torch", dtype="float32", device="cpu")

    import torch

    assert traj.backend == "torch"
    assert isinstance(traj.kx, torch.Tensor)
    assert traj.kx.dtype == torch.float32
    assert traj.kx.device.type == "cpu"


def test_backend_conversion_cupy(sample_trajectory):
    """Test that `to()` converts arrays to the correct cupy backend."""
    traj = sample_trajectory
    traj.to("cupy", dtype=np.float32)

    import cupy as cp

    assert traj.backend == "cupy"
    assert isinstance(traj.kx, cp.ndarray)
    assert traj.kx.dtype == cp.float32


def test_automatic_axis_generation(sample_trajectory):
    """Test that missing axes are generated automatically."""
    traj = Trajectory(
        kx=np.random.rand(1, 1, 3, 2, 5),
        ky=np.random.rand(1, 1, 3, 2, 5),
        ncontrast=3,
        ntime=2,
        nz=3,
    )

    # Automatically generated axes
    assert traj.contrast_axis is not None
    assert traj.time_axis is not None
    assert traj.kz is not None

    assert traj.contrast_axis.shape == (1, 3, 1, 1, 1)
    assert traj.time_axis.shape == (2, 1, 1, 1, 1)
    assert traj.kz.shape == (1, 1, 3, 1, 1)


def test_shape_compatibility():
    """Test that shape compatibility checks are done properly."""
    kx = np.random.rand(1, 1, 3, 2, 5)
    ky = np.random.rand(1, 1, 3, 2, 5)
    contrast_axis = np.random.rand(1, 3, 1, 1, 1)
    time_axis = np.random.rand(2, 1, 1, 1, 1)

    traj = Trajectory(
        kx=kx,
        ky=ky,
        contrast_axis=contrast_axis,
        time_axis=time_axis,
        ncontrast=3,
        ntime=2,
    )

    # Check that the trajectory's shape is consistent with broadcasting
    assert traj.shape == (2, 3, 1, 2, 5)  # Broadcasted shape of all attributes


def test_index_matrix(sample_trajectory):
    """Test that `index_matrix` returns the correct (n, 2) index matrix."""
    traj = sample_trajectory
    index_matrix = traj.index_matrix

    # Check that the shape of index_matrix matches the expected total number of elements
    total_elements = np.prod(traj.shape)
    assert index_matrix.shape == (total_elements, 2)

    # The first column should contain raveled stack indexes (i.e., linear indices)
    assert np.all(
        index_matrix[:, 0]
        == np.ravel_multi_index(np.indices(traj.shape), traj.shape).flatten()
    )

    # The second column should contain sequential raveled trajectory indices
    assert np.all(index_matrix[:, 1] == np.arange(total_elements))


def test_value_matrix(sample_trajectory):
    """Test that `value_matrix` returns the correct (n, m) value matrix."""
    traj = sample_trajectory
    value_matrix = traj.value_matrix

    # Check that the shape of value_matrix matches the expected total number of elements
    total_elements = np.prod(traj.shape)
    assert value_matrix.shape == (total_elements, 5)  # Time, contrast, kz, ky, kx

    # The first few columns should contain stack axis values (time_axis, contrast_axis, kz)
    assert value_matrix[:, 0].shape == (total_elements,)  # time_axis
    assert value_matrix[:, 1].shape == (total_elements,)  # contrast_axis
    assert value_matrix[:, 2].shape == (total_elements,)  # kz
    assert value_matrix[:, 3].shape == (total_elements,)  # ky
    assert value_matrix[:, 4].shape == (total_elements,)  # kx
