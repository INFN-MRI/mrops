mrops: Matrix-Free MRI Reconstruction Operators
==============================================

*mrops* is a lightweight Python library for matrix-free MRI reconstruction.  
Built on top of `SigPy <https://github.com/mikgroup/sigpy>`_, *mrops* provides  
drop-in replacements for SigPy operators such as `NUFFT` and `FFT`, leveraging  
state-of-the-art implementations like `FINUFFT <https://finufft.readthedocs.io>`_  
and `cuFINUFFT <https://github.com/flatironinstitute/cufinufft>`_ for optimal  
performance.  

Additionally, *mrops* offers seamless interoperability with  
`scipy.sparse.linalg.LinearOperator` and `cupyx.scipy.sparse.linalg.LinearOperator`,  
enabling integration with established optimization solvers like Conjugate Gradient (CG)  
and Least Squares Minimal Residual (LSMR). It also supports PyTorch autograd,  
allowing its operators to be used in deep learning-based image reconstruction frameworks  
such as `deepinv <https://github.com/deepinv/deepinv>`_ and  
`DeepInPy <https://github.com/deepinpy/deepinpy>`_.  

Key Features
------------

- **Optimized MRI Reconstruction**: Drop-in replacements for SigPy's `NUFFT`, `FFT`, etc.
- **High-Performance Implementations**: Uses `finufft` and `cufinufft` for acceleration.
- **Interoperability with SciPy & CuPy**: Compatible with `LinearOperator`-based solvers.
- **PyTorch Support**: Enables deep learning-based reconstruction workflows.
- **GPU Acceleration**: Leverages CUDA-based libraries for efficient computations.

Installation
------------

You can install *mrops* via pip:

.. code-block:: bash

    pip install mrops

To use GPU acceleration, make sure you have `cuFINUFFT` installed:

.. code-block:: bash

    pip install cufinufft

Getting Started
---------------

Here's a quick example demonstrating how to use *mrops*:

.. code-block:: python

    import mrops
    import sigpy as sp
    import numpy as np

    # Define an MRI sampling pattern
    shape = (256, 256)
    mask = np.random.rand(*shape) < 0.3  # Simulated undersampling mask

    # Create an NUFFT operator using mrops
    nufft_op = mrops.NUFFT(shape, mask)

    # Apply forward and adjoint operations
    img = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    kspace = nufft_op * img  # Forward NUFFT
    img_recon = nufft_op.H * kspace  # Adjoint NUFFT

    # Use in a SciPy solver (e.g., Conjugate Gradient)
    from scipy.sparse.linalg import cg
    img_cg, _ = cg(nufft_op.H @ nufft_op, nufft_op.H @ kspace)

    print("Reconstruction complete!")

Contributing
------------

We welcome contributions! If you find a bug, have a feature request,  
or want to contribute, please open an issue or submit a pull request  
on our `GitHub repository <https://github.com/yourusername/mrops>`_.  

License
-------

*mrops* is released under the MIT License.

