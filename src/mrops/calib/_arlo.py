"""ARLO-based R2* mapping."""

__all__ = ["arlo"]


from numpy.typing import NDArray

import torch

from mrinufft._array_compat import with_torch


@with_torch
def arlo(input: NDArray[float], te: NDArray[float]) -> NDArray[float]:
    """
    Compute the R2 map (in Hz) using the ARLO algorithm.

    Parameters
    ----------
    input : NDArray[float]
        A multi-echo dataset of arbitrary dimension.
        The first dimension should correspond to the echo times.
    te : NDArray[float]
        Sequence of TE values (in milliseconds).

    Returns
    -------
    NDArray[float]
        The R2 map. Returns an empty tensor if only one echo is provided.

    Notes
    -----
    If you use this function, please cite:

    Pei M, Nguyen TD, Thimmappa ND, Salustri C, Dong F, Cooper MA, Li J,
    Prince MR, Wang Y. Algorithm for fast monoexponential fitting based
    on Auto-Regression on Linear Operations (ARLO) of data.
    Magn Reson Med. 2015 Feb;73(2):843-50. doi: 10.1002/mrm.25137.
    Epub 2014 Mar 24. PubMed PMID: 24664497;
    PubMed Central PMCID: PMC4175304.
    """
    te = te * 1e-3  # [ms[ -> [s]

    # tranpose input to Fortran order for computation
    _order = tuple(range(input.ndim))[::-1]
    y = abs(input).permute(*_order)  # (te, z, y, x) -> (x, y, z, te)
    y = y.contiguous()

    nte = te.numel()
    if nte < 2:
        return torch.tensor([])

    sz = y.shape
    if sz[-1] != nte:
        raise ValueError(f"Last dimension of y has size {sz[-1]}, expected {nte}")

    yy = torch.zeros(sz[:-1], dtype=y.dtype, device=y.device)
    yx = torch.zeros_like(yy)
    beta_yx = torch.zeros_like(yy)
    beta_xx = torch.zeros_like(yy)

    for j in range(nte - 2):
        alpha = (te[j + 2] - te[j]) ** 2 / (2 * (te[j + 1] - te[j]))
        tmp = (
            2 * te[j + 2] ** 2
            - te[j] * te[j + 2]
            - te[j] ** 2
            + 3 * te[j] * te[j + 1]
            - 3 * te[j + 1] * te[j + 2]
        ) / 6
        beta = tmp / (te[j + 2] - te[j + 1])
        gamma = tmp / (te[j + 1] - te[j])

        y1 = (
            y[..., j] * (te[j + 2] - te[j] - alpha + gamma)
            + y[..., j + 1] * (alpha - beta - gamma)
            + y[..., j + 2] * beta
        )
        x1 = y[..., j] - y[..., j + 2]

        yy += y1 * y1
        yx += y1 * x1
        beta_yx += beta * y1 * x1
        beta_xx += beta * x1 * x1

    r2 = (yx + beta_xx) / (beta_yx + yy)
    r2[torch.isnan(r2)] = 0
    r2[torch.isinf(r2)] = 0

    # tranpose output back to C order
    _order = tuple(range(r2.ndim))[::-1]
    r2 = r2.permute(*_order)  # (x, y, z, te) -> (te, z, y, x)

    return r2.contiguous()
