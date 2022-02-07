#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

from .__about__ import __author__, __description__, __version__
from .log import *


def has_torch():
    try:
        import torch

        return True
    except ModuleNotFoundError:
        return False


# HAS_CUDA = False
HAS_TORCH = has_torch()


def _has_cuda():
    try:
        import torch

        return torch.cuda.is_available()
    except ModuleNotFoundError:
        return False


HAS_CUDA = _has_cuda()


_nannos_device = "cpu"


def use_gpu():
    global _nannos_device
    if not HAS_TORCH:
        _nannos_device = "cpu"
        log.info("pytorch not found. Cannot use GPU.")
    elif not HAS_CUDA:
        _nannos_device = "cpu"
        log.info("cuda not found. Cannot use GPU.")
    else:
        _nannos_device = "cuda"
        log.info("Using GPU.")


def _delvar(VAR):
    if VAR in globals():
        del globals()[VAR]


def set_backend(backend):
    """Set the numerical backend.

    Parameters
    ----------
    backend : str
        Either ``numpy``, ``scipy``, ``autograd``, ``torch`` or ``jax``.


    """

    import importlib
    import sys

    global _NUMPY
    global _SCIPY
    global _AUTOGRAD
    global _JAX
    global _TORCH
    if backend == "autograd":
        log.info("Setting autograd backend")
        _AUTOGRAD = True
        _delvar("_JAX")
        _delvar("_TORCH")
        _delvar("_SCIPY")
    elif backend == "scipy":
        log.info("Setting scipy backend")
        _SCIPY = True
        _delvar("_JAX")
        _delvar("_TORCH")
        _delvar("_AUTOGRAD")
    elif backend == "jax":
        log.info("Setting jax backend")
        _JAX = True
        _delvar("_SCIPY")
        _delvar("_TORCH")
        _delvar("_AUTOGRAD")
    elif backend == "torch":
        _TORCH = True
        log.info("Setting torch backend")
        _delvar("_SCIPY")
        _delvar("_JAX")
        _delvar("_AUTOGRAD")
    elif backend == "numpy":
        _NUMPY = True
        log.info("Setting numpy backend")
        _delvar("_SCIPY")
        _delvar("_JAX")
        _delvar("_AUTOGRAD")
        _delvar("_TORCH")
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Please choose between 'numpy', 'scipy', 'jax', 'torch' and 'autograd'."
        )

    import nannos

    importlib.reload(nannos)

    its = [s for s in sys.modules.items() if s[0].startswith("nannos")]
    for k, v in its:
        importlib.reload(v)


def get_backend():

    if "_SCIPY" in globals():
        return "scipy"
    elif "_AUTOGRAD" in globals():
        return "autograd"
    elif "_JAX" in globals():
        return "jax"
    elif "_TORCH" in globals():
        return "torch"
    else:
        return "numpy"


def grad(f):
    raise NotImplementedError(f"grad is not implemented for {BACKEND} backend.")


if "_SCIPY" in globals():

    import numpy

    backend = numpy
elif "_AUTOGRAD" in globals():
    from autograd import grad, numpy

    backend = numpy
elif "_JAX" in globals():
    _JAX
    from jax.config import config

    config.update("jax_enable_x64", True)
    from jax import grad, numpy

    backend = numpy
elif "_TORCH" in globals():
    if HAS_TORCH:
        import numpy
        import torch

        backend = torch

        def _array(a, **kwargs):
            if isinstance(a, backend.Tensor):
                return a.to(_nannos_device)
            else:
                return backend.tensor(a, **kwargs).to(_nannos_device)

        backend.array = _array

        def grad(f):
            def df(x):
                _x = x.clone().detach().requires_grad_(True)
                return backend.autograd.grad(f(_x), _x, allow_unused=True)[0]

            return df

    else:
        log.info("pytorch not found. Falling back to default backend.")
        set_backend("numpy")
else:
    import numpy

    backend = numpy

# TODO: support jax since it is faster than autograd
# jax does not support numpy.linalg.eig
# for autodif wrt eigenvectors yet.
# see: https://github.com/google/jax/issues/2748

BACKEND = get_backend()

from .constants import *
from .excitation import *
from .lattice import *
from .layers import *
from .parallel import *
from .sample import *
from .simulation import *
from .utils import *
