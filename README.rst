pygrog: A PyTorch-based Package for GROG Interpolation
======================================================

PyGROG is a lightweight PyTorch library for implementing GRAPPA operator gridding (GROG).
 
This package enables efficient interpolation of non-Cartesian MRI data onto Cartesian grids using GROG operator
and faster iterative reconstruction. 

Key Features
------------

- **Trajectory Pre-processing**: Pre-sort k-space trajectory for fast non-Cartesian dataset interpolation at runtime.
- **GRAPPA Training**: Set up and train GROG interpolation kernel.
- **Interpolation**: Apply GROG operator to new non-Cartesian datasets for interpolation.
- **Fast zero-filling and indexing**: Convert interpolated sparse data to dense Cartesian grids and vice-versa.
- **Expanded Fourier Model**: Expand signal model to include subspace projection and off-resonance modeling.

In addition:

- **Calibration**: Extract low-resolution k-space data or synthesize calibration data based on NLINV.
- **Interoperability with SciPy & CuPy**: Compatible with LinearOperator-based solvers.

Installation
------------

You can install PyGROG via pip:

.. code-block:: bash

    pip install pygrog

Getting Started
---------------

Here's a quick example demonstrating how to use PyGROG:

.. code-block:: python

    import pygrog
    import numpy as np


Contributing
------------

We welcome contributions! If you find a bug, have a feature request,  
or want to contribute, please open an issue or submit a pull request  
on our `GitHub repository <https://github.com/INFN-MRI/pygrog>`_.  

License
-------

PyGROG is released under the MIT License.

