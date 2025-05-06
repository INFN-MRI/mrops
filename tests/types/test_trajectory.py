"""Test trajectory container."""

import numpy as np
import pytest

from pygrog.types import Trajectory


@pytest.fixture
def kspace_coords_2d():
    nx, ny = 4, 4
    x = np.linspace(-0.5, 0.5, nx)
    y = np.linspace(-0.5, 0.5, ny)
    kx, ky = np.meshgrid(x, y, indexing="ij")
    return kx, ky


@pytest.fixture
def kspace_coords_3d():
    nx, ny, nz = 4, 4, 4
    x = np.linspace(-0.5, 0.5, nx)
    y = np.linspace(-0.5, 0.5, ny)
    z = np.linspace(-0.5, 0.5, nz)
    kx, ky, kz = np.meshgrid(x, y, z, indexing="ij")
    return kx, ky, kz


def test_2d_no_stack(kspace_coords_2d):
    kx, ky = kspace_coords_2d
    traj = Trajectory(ndim=2, nx=4, ny=4, kx=kx, ky=ky)

    assert traj.indexes.shape[0] == 2
    assert traj.values.shape[0] == 2
    assert traj.values.shape[1] == kx.size


def test_2d_with_stack(kspace_coords_2d):
    kx, ky = kspace_coords_2d
    slice_axis = np.arange(3)
    contrast_axis = np.arange(2)

    traj = Trajectory(
        ndim=2,
        nx=4,
        ny=4,
        kx=kx,
        ky=ky,
        slice_axis=slice_axis,
        contrast_axis=contrast_axis,
    )

    n_stack = len(slice_axis) * len(contrast_axis)
    n_spatial = kx.size
    assert traj.indexes.shape == (2, n_stack * n_spatial)
    assert traj.values.shape == (2, n_stack * n_spatial)


def test_3d_with_stack(kspace_coords_3d):
    kx, ky, kz = kspace_coords_3d
    time_axis = np.arange(3)

    traj = Trajectory(
        ndim=3,
        nx=4,
        ny=4,
        nz=4,
        kx=kx,
        ky=ky,
        kz=kz,
        time_axis=time_axis,
    )

    n_stack = len(time_axis)
    n_spatial = kx.size
    assert traj.indexes.shape == (2, n_stack * n_spatial)
    assert traj.values.shape == (2, n_stack * n_spatial)  # kx, ky, kz


def test_index_value_consistency(kspace_coords_2d):
    kx, ky = kspace_coords_2d
    time_axis = np.arange(2)
    traj = Trajectory(
        ndim=2,
        nx=4,
        ny=4,
        kx=kx,
        ky=ky,
        time_axis=time_axis,
        nframes=2,
    )

    # index[1] gives spatial index
    # values[:, i] should equal the corresponding raveled coordinates
    spatial = traj.indexes[1]
    for i, s in enumerate(spatial):
        np.testing.assert_allclose(
            traj.values[:, i], [traj.kx.ravel()[s], traj.ky.ravel()[s]]
        )


def test_caching_of_indexes_and_values(kspace_coords_2d):
    kx, ky = kspace_coords_2d
    traj = Trajectory(ndim=2, nx=4, ny=4, kx=kx, ky=ky)
    # Access triggers caching
    _ = traj.indexes
    _ = traj.values
    assert traj._indexes is not None
    assert traj._values is not None


def test_single_element_stack_axes(kspace_coords_2d):
    kx, ky = kspace_coords_2d
    traj = Trajectory(
        ndim=2,
        nx=4,
        ny=4,
        kx=kx,
        ky=ky,
        nslices=1,
        ncontrasts=1,
        nframes=1,
        slice_axis=np.array([0]),
        contrast_axis=np.array([0]),
        time_axis=np.array([0]),
    )

    n_spatial = kx.size
    assert traj.indexes.shape == (2, n_spatial)
    assert traj.values.shape == (2, n_spatial)
