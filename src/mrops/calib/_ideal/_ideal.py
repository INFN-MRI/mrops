"""IDEAL algorithm for iterative fat-water separation with joint fieldmap estimation."""

__all__ = ["ideal", "_FatWater"]


import numpy as np
from numpy.typing import NDArray

from .._sigpy import get_device


def ideal(): ...


def _FatWater(te: NDArray[float], B0: float) -> NDArray[complex]:
    """
    Get Fat Water basis.

    Parameters
    ----------
    te : NDArray[float]
        Echo times in ``[s]``.
    B0 : float
        Field strength in ``[T]``.

    Returns
    -------
    NDArray[complex]
        Water-Fat basis at given field strength.

    """
    device = get_device(te)
    xp = device.xp

    # Fat-water matrix
    ne = len(te)
    te = te.reshape(ne, 1)

    with device:
        d = np.asarray(
            [5.29, 5.19, 4.20, 2.75, 2.24, 2.02, 1.60, 1.30, 0.90], dtype=xp.float32
        )
        fat = xp.zeros_like(te, dtype=xp.complex64)
        freq = xp.zeros(len(d), dtype=xp.float32)

    # Number of double bond, water model
    NDB, H2O = 2.5, 4.7

    # Mark's heuristic formulas
    CL = 16.8 + 0.25 * NDB
    NDDB = 0.093 * NDB**2

    # Gavin's formulas (number of protons per molecule)
    awater = 2.0
    ampl = np.zeros(len(d))
    ampl[0] = NDB * 2
    ampl[1] = 1
    ampl[2] = 4
    ampl[3] = NDDB * 2
    ampl[4] = 6
    ampl[5] = (NDB - NDDB) * 4
    ampl[6] = 6
    ampl[7] = (CL - 4) * 6 - NDB * 8 + NDDB * 2
    ampl[8] = 9

    # Scaling
    awater = awater / 2
    ampl = ampl / ampl.sum()

    # Time evolution matrix
    water = (te * 0.0 + awater).astype(fat.dtype)
    larmor = 42.57747892 * B0  # Larmor frequency (MHz)

    for j in range(len(d)):
        freq[j] = larmor * (d[j] - H2O)  # relative to water
        fat += ampl[j] * np.exp(2 * np.pi * 1j * freq[j] * te)

    return xp.concatenate((water, fat), axis=-1)


class _FieldModel:
    def __init__(self): ...


class PhaseModel: ...
