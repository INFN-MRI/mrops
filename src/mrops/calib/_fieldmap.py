"""Field map calibration."""

__all__ = ["calc_fieldmap"]

import warnings

import numpy as np
import torch

from numpy.typing import NDArray
from skimage.restoration import unwrap_phase
from mrinufft._array_compat import with_numpy, with_torch

from ..base._fftc import fft, ifft
from .._sigpy import get_device
from ._nlinv import kspace_filter


@with_torch
def calc_fieldmap(
    input: NDArray[complex],
    te: NDArray[float],
    ref_vol: int = 0,
    kw: float = 200.0,
    ell: int = 32,
) -> NDArray[float]:
    """
    Compute the B0 field map from multi-echo MRI phase images.

    Parameters
    ----------
    input : NDArray[complex]
        Complex-valued multi-echo MRI data of shape ``(nte, nz, ny, nx)`` for 3D data
        or ``(nte, nx, ny)`` for single-slice 2D data, where ``nte`` is the number of echo times.
    te : NDArray[float]
        A 1D array of echo times (in ``[ms]``) of shape ``(nte,)``.

    Returns
    -------
    NDArray[float]
        Estimated B0 field map in ``[Hz]``, with shape ``(nz, ny, nx)`` for 3D input or ``(nx, ny)`` for 2D input.

    Raises
    ------
    ValueError
        If fewer than two echo times are provided.

    Notes
    -----
    - If only two echoes are available, B0 is computed using the simple phase difference method:

      .. math:: B_0 = \\frac{\\text{uphase}}{2\\pi (TE_2 - TE_1)}

    - If more than two echoes are available, B0 is estimated as the voxelwise slope of a linear
      least squares fit of `uphase` against `te`.

    """
    if len(te) == 1:
        raise ValueError(f"At least 2 different TEs are required, found {len(te)}")

    # low pass filter
    smooth = True
    ndim = input.ndim - 1
    if smooth:
        n = np.max(input.shape[-ndim:])
        filt = kspace_filter(input.shape[-ndim:], kw / n**2, ell, get_device(input))
        filt = torch.as_tensor(filt, dtype=input.dtype, device=input.device)
        input = fft(input, axes=tuple(range(-ndim, 0)))
        input = filt * input
        input = ifft(input, axes=tuple(range(-ndim, 0)))
    # uphase = romeo_like_unwrap(input, te, ref_vol, False, kw, ell)
    uphase = torch.stack(
        [spatial_unwrap(ph, False, 0.0, 0) for ph in input.angle()], axis=0
    )

    if len(te) == 2:
        # Two echoes: simple phase difference method
        b0 = (uphase[1] - uphase[0]) / (2 * torch.pi * (te[1] - te[0]))
    else:
        mag = torch.abs(
            input
        )  # Ignore first echo for consistency with phase computation
        weights = mag / torch.sum(mag, axis=0, keepdims=True)  # Normalize across echoes

        # Design matrix A
        A = torch.vstack([te, torch.ones_like(te)]).T

        # Apply weights to A and uphase
        W = torch.sqrt(weights).reshape(len(te), -1)  # Weight matrix (flattened)
        Aw = A * W.T[:, :, None]  # Broadcast weights onto A
        bw = uphase.reshape(len(te), -1) * W  # Weight uphase
        bw = bw.T

        # Solve weighted least squares for each voxel
        slope, _ = torch.linalg.lstsq(Aw, bw, rcond=None)[0].T

        # Extract B0 map (first coefficient) and reshape back
        b0 = (
            slope.reshape(uphase.shape[1:]) / 2 / torch.pi
        )  # Reshape back to original spatial dimensions

    return 1e3 * b0


# %% subroutines
def romeo_like_unwrap(
    input: NDArray[float],
    te: NDArray[float],
    ref_vol: int = 0,
    smooth: bool = False,
    kw: float = 220.0,
    ell: int = 16,
) -> NDArray[float]:
    """
    Perform a ROMEO-like unwrap, i.e., a spatial unwrap across
    a reference volume followed by temporal unwrap.

    For the moment, we do not use the same quality-guided unwrap
    algorithm as ROMEO, but the scikit-image built-in unwrapping algorithm.

    Parameters
    ----------
    phase : NDArray[float]
        Input wrapped phase of shape ``(nvol, nz, ny, nx)``.
    te : NDArray[float]
        A 1D array of echo times (in ``[ms]``) of shape ``(nte,)``.
    ref_vol : int, optional
        Reference volume for spatial unwrapping. The default is 0.
    smooth : bool, optional
        Toggle smooting of input phase. The default is ``False``.
    kw : float, optional
        Smoothing filter width. The default is ``220.0``.
    ell : int, optional
        Smoothing filter order (as in Sobolev norm). The default is ``32``.

    Returns
    -------
    NDArray[float]
        Unrwapped phase of the same shape as input.

    """
    phase = input.angle()
    ref_uphase = spatial_unwrap(phase[ref_vol], smooth, kw, ell)

    # now use reference volume to perform temporal unwrapping
    return temporal_unwrap(phase, ref_uphase, te, ref_vol)


def spatial_unwrap(phase, smooth, kw, ell):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        return _unwrap(phase, smooth, kw, ell)


@with_numpy
def _unwrap(phase, smooth, kw, ell):
    uphase = unwrap_phase(phase).astype(phase.dtype)

    return uphase


def temporal_unwrap(phase_series, ref_unwrapped, te, ref_idx):
    unwrapped_phase = torch.full_like(phase_series, torch.nan)
    unwrapped_phase[ref_idx] = ref_unwrapped  # Set reference echo

    # Create a queue of echoes to unwrap, starting with the reference
    queue = [ref_idx]

    while queue:
        current_idx = queue.pop(0)  # Get the next echo to process
        current_phase = unwrapped_phase[current_idx]  # Unwrapped phase at this echo
        current_te = te[current_idx]

        # Process neighbors (previous and next echoes)
        for neighbor_idx in [current_idx - 1, current_idx + 1]:
            if (
                0 <= neighbor_idx < len(te)
                and torch.isnan(unwrapped_phase[neighbor_idx]).any()
            ):
                neighbor_te = te[neighbor_idx]
                raw_phase = phase_series[neighbor_idx]

                # Apply temporal unwrapping equation
                phase_correction = (
                    2
                    * np.pi
                    * torch.round(
                        (raw_phase - current_phase * (neighbor_te / current_te))
                        / (2 * torch.pi)
                    )
                )
                unwrapped_phase[neighbor_idx] = raw_phase - phase_correction

                queue.append(neighbor_idx)  # Add newly unwrapped echo to queue

    return unwrapped_phase
