"""Test trajectory container."""

import itertools
import numpy as np

import pytest

from pygrog.types import Trajectory


# === Fixtures === #
@pytest.fixture
def kspace_2d():
    kx = np.random.randn(4, 128)
    ky = np.random.randn(4, 128)
    return kx, ky


@pytest.fixture
def kspace_3d():
    kx = np.random.randn(4, 128)
    ky = np.random.randn(4, 128)
    kz = np.random.randn(4, 128)
    return kx, ky, kz


@pytest.fixture
def stack_axes():
    return {
        "time_axis": np.arange(2),
        "contrast_axis": np.arange(3),
        "slice_axis": np.arange(4),
    }


# === Tests for 2D Trajectories === #
def test_basic_2d_trajectory(kspace_2d):
    kx, ky = kspace_2d
    traj = Trajectory(ndim=2, nx=128, ny=128, kx=kx, ky=ky)

    assert traj.ndim == 2
    assert traj.kx.shape == (1, 1, 1, 4, 128)
    assert traj.grid_shape == (128, 128)
    assert traj.kz is None
    assert "2D trajectory" in str(traj)


def test_2d_with_stack_axes(kspace_2d, stack_axes):
    kx, ky = kspace_2d
    traj = Trajectory(ndim=2, nx=128, ny=128, kx=kx, ky=ky, **stack_axes)

    coords, indexes = traj.coords_and_indexes
    assert coords.shape[1] == 2
    assert indexes.shape[1] == 3
    assert traj.stack_shape == (2, 3, 4)


def test_hybrid_trajectory_ndim3(kspace_2d, stack_axes):
    kx, ky = kspace_2d
    traj = Trajectory(
        ndim=3, nx=128, ny=128, kx=kx, ky=ky, slice_axis=stack_axes["slice_axis"]
    )
    assert traj.ndim == 3
    assert traj._hybrid_trajectory is True
    coords, indexes = traj.coords_and_indexes
    assert coords.shape[1] == 3
    assert traj.nz == 4


# === Tests for 3D Trajectories === #
def test_basic_3d_trajectory(kspace_3d):
    kx, ky, kz = kspace_3d
    traj = Trajectory(ndim=3, nx=128, ny=128, nz=64, kx=kx, ky=ky, kz=kz)

    assert traj.ndim == 3
    assert traj.kz is not None
    assert traj.grid_shape == (64, 128, 128)

    coords, indexes = traj.coords_and_indexes
    assert coords.shape[1] == 3


def test_3d_with_stack_axes(kspace_3d, stack_axes):
    kx, ky, kz = kspace_3d
    traj = Trajectory(
        ndim=3,
        nx=128,
        ny=128,
        nz=64,
        kx=kx,
        ky=ky,
        kz=kz,
        time_axis=stack_axes["time_axis"],
        contrast_axis=stack_axes["contrast_axis"],
    )

    assert traj.stack_shape[:2] == (2, 3)
    coords, indexes = traj.coords_and_indexes
    assert coords.shape[1] == 3
    assert indexes.shape[1] == 2


# === Internal Logic Tests === #
def test_coords_and_indexes_match_shapes(kspace_2d, stack_axes):
    kx, ky = kspace_2d
    traj = Trajectory(ndim=2, nx=128, ny=128, kx=kx, ky=ky, **stack_axes)

    coords, indexes = traj.coords_and_indexes
    assert coords.shape[0] == indexes.shape[0]


def test_stack_shape_inference(kspace_2d, stack_axes):
    kx, ky = kspace_2d
    traj = Trajectory(ndim=2, nx=128, ny=128, kx=kx, ky=ky, **stack_axes)
    assert traj.stack_shape == (2, 3, 4)


@pytest.mark.parametrize("axes_combo", list(itertools.product([True, False], repeat=3)))
def test_partial_stack_axes_combinations(kspace_2d, axes_combo):
    include_time, include_contrast, include_slice = axes_combo
    kx, ky = kspace_2d

    kwargs = {"ndim": 2, "nx": 128, "ny": 128, "kx": kx, "ky": ky}
    expected_stack_shape = []

    if include_time:
        kwargs["time_axis"] = np.arange(2)
        expected_stack_shape.append(2)
    if include_contrast:
        kwargs["contrast_axis"] = np.arange(3)
        expected_stack_shape.append(3)
    if include_slice:
        kwargs["slice_axis"] = np.arange(4)
        expected_stack_shape.append(4)

    traj = Trajectory(**kwargs)

    assert traj.ndim == 2
    assert traj.kx.shape == (1, 1, 1, 4, 128)
    assert traj.grid_shape == (128, 128)
    assert traj.stack_shape == tuple(expected_stack_shape)

    coords, indexes = traj.coords_and_indexes
    assert coords.shape[1] == 2  # 2D coords
    if len(expected_stack_shape):
        assert coords.shape[0] == indexes.shape[0]
        assert indexes.shape[1] == len(expected_stack_shape)


@pytest.mark.parametrize("axes_combo", list(itertools.product([True, False], repeat=2)))
@pytest.mark.parametrize("hybrid", [True, False])
def test_partial_stack_axes_combinations_3d(kspace_3d, axes_combo, hybrid):
    include_time, include_contrast = axes_combo
    kx, ky, kz = kspace_3d

    kwargs = {"ndim": 3, "nx": 64, "ny": 64, "kx": kx, "ky": ky}
    expected_stack_shape = []

    # Use either kz (Fourier stack) or slice_axis (hybrid stack) for the z-dimension
    if hybrid:
        kwargs["slice_axis"] = np.arange(5)
        expected_stack_shape.append(5)
    else:
        kwargs["nz"] = 5
        kwargs["kz"] = kz

    if include_time:
        kwargs["time_axis"] = np.arange(2)
        expected_stack_shape.insert(0, 2)
    if include_contrast:
        kwargs["contrast_axis"] = np.arange(3)
        if include_time:
            expected_stack_shape.insert(1, 3)
        else:
            expected_stack_shape.insert(0, 3)

    traj = Trajectory(**kwargs)

    assert traj.ndim == 3
    assert traj.kx.ndim >= 2
    if hybrid:
        assert traj._hybrid_trajectory is True
    else:
        assert traj.kz is not None

    coords, indexes = traj.coords_and_indexes
    assert coords.shape[1] == 3  # 3D coords
    if len(expected_stack_shape):
        assert coords.shape[0] == indexes.shape[0]
        assert indexes.shape[1] == len(expected_stack_shape)


def test_shape_property(kspace_2d, stack_axes):
    kx, ky = kspace_2d
    traj = Trajectory(ndim=2, nx=128, ny=128, kx=kx, ky=ky, **stack_axes)
    assert traj.shape == (2, 3, 4, 128, 128)


def test_lazy_evaluation_caching(kspace_2d, stack_axes):
    kx, ky = kspace_2d
    traj = Trajectory(ndim=2, nx=128, ny=128, kx=kx, ky=ky, **stack_axes)

    coords1, indexes1 = traj.coords_and_indexes
    coords2, indexes2 = traj.coords_and_indexes  # should be cached
    np.testing.assert_array_equal(coords1, coords2)
    np.testing.assert_array_equal(indexes1, indexes2)


def test_str_repr(kspace_2d, stack_axes):
    kx, ky = kspace_2d
    traj = Trajectory(ndim=2, nx=128, ny=128, kx=kx, ky=ky, **stack_axes)
    assert isinstance(str(traj), str)
    assert "trajectory" in str(traj)
