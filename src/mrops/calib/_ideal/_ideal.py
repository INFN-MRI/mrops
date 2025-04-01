"""IDEAL algorithm."""

import numpy as np
import scipy.ndimage
import skimage.restoration
import torch
import torch.nn.functional as F
from _ideal_op import IdealOp
from _ideal_reg import IrgnmBase, LSTSQ

# ---------------------------
# Unswap and Median Filtering Functions
# ---------------------------


def downsample_image(image, factor, mode="trilinear"):
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
    new_size = [int(dim / factor) for dim in tensor.shape[2:]]
    down = F.interpolate(tensor, size=new_size, mode=mode, align_corners=False)
    return down.squeeze().numpy()


def upsample_image(image, factor, mode="trilinear"):
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
    new_size = [int(dim * factor) for dim in tensor.shape[2:]]
    up = F.interpolate(tensor, size=new_size, mode=mode, align_corners=False)
    return up.squeeze().numpy()


def unswap(psi, downsample_factor=4):
    B0 = np.real(psi).squeeze()
    B0_ds = downsample_image(B0, factor=downsample_factor)
    B0_unwrapped_ds = np.array(
        [skimage.restoration.unwrap_phase(B0_ds[i]) for i in range(B0_ds.shape[0])]
    )
    B0_unwrapped = upsample_image(B0_unwrapped_ds, factor=downsample_factor)
    psi_updated = B0_unwrapped[np.newaxis, ...] + 1j * np.imag(psi.squeeze())
    return psi_updated


def median_filter_image(image, size):
    return scipy.ndimage.median_filter(image, size=size)


# ---------------------------
# Full IDEAL Algorithm
# ---------------------------


def ideal_algorithm(psi_init, te, data, A, **kwargs):
    psi = psi_init.copy()

    for iter in range(kwargs.get("maxit_outer", 10)):
        print(f"Outer Iteration {iter+1}")

        solver = IdealIRGNM(te, data, psi, A, **kwargs)
        psi = solver.solve()

        op = IdealOp(te, A, **kwargs)
        residual = data - op._compute_forward(psi).asarray()
        print("Residual norm:", np.linalg.norm(residual))

        if iter < kwargs.get("maxit_outer", 10) - 1:
            if kwargs.get("unwrap_flag", True):
                psi = unswap(psi, kwargs.get("downsample_factor", 4))
            if kwargs.get("smooth_field", True):
                B0_filtered = median_filter_image(
                    np.real(psi), size=kwargs.get("median_filter_size", (3, 3, 3))
                )
                psi = B0_filtered[np.newaxis, ...] + 1j * np.imag(psi.squeeze())

    return psi
