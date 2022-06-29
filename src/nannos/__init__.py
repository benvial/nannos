#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

import os

from .__about__ import __author__, __description__, __version__
from .__about__ import data as _data
from .log import *

available_backends = ["numpy", "scipy", "autograd", "jax", "torch"]


def print_info():
    print(f"nannos v{__version__}")
    print("=============")
    print(__description__)
    print(f"Author: {__author__}")
    print(f"Licence: {_data['License']}")


def has_torch():
    try:
        import torch

        return True
    except ModuleNotFoundError:
        return False


def has_jax():
    try:
        import jax

        return True
    except ModuleNotFoundError:
        return False


def _has_cuda():
    try:
        import torch

        return torch.cuda.is_available()
    except ModuleNotFoundError:
        return False


HAS_TORCH = has_torch()
HAS_CUDA = _has_cuda()

HAS_JAX = has_jax()

if HAS_TORCH:
    available_backends.append("torch")

if HAS_JAX:
    available_backends.append("jax")


def use_gpu(boolean):
    global DEVICE
    global _CPU_DEVICE
    global _GPU_DEVICE

    if boolean:

        if BACKEND not in ["torch"]:
            logger.debug(f"Cannot use GPU with {BACKEND} backend.")
            _delvar("_GPU_DEVICE")
            _CPU_DEVICE = True
        else:
            if not HAS_TORCH:
                logger.warning("pytorch not found. Cannot use GPU.")
                _delvar("_GPU_DEVICE")
                _CPU_DEVICE = True
            elif not HAS_CUDA:
                logger.warning("cuda not found. Cannot use GPU.")
                _delvar("_GPU_DEVICE")
                _CPU_DEVICE = True
            else:
                DEVICE = "cuda"
                logger.debug("Using GPU.")
                _delvar("_CPU_DEVICE")
                _GPU_DEVICE = True
    else:
        _CPU_DEVICE = True
        _delvar("_GPU_DEVICE")
        DEVICE = "cpu"
        logger.debug("Using CPU.")
    _reload_package()


def jit(fun, **kwargs):
    if BACKEND == "jax":
        from jax import jit

        return jit(fun, **kwargs)
    else:
        return fun


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

    global _NUMPY
    global _SCIPY
    global _AUTOGRAD
    global _JAX
    global _TORCH
    global _FORCE_BACKEND

    _FORCE_BACKEND = 1

    if backend == get_backend():
        return
    #
    # _backend_env_var = os.environ.get("NANNOS_BACKEND")
    # if _backend_env_var is not None:
    #     if _backend_env_var in available_backends:
    #         if backend != _backend_env_var:
    #             # _delvar("_FORCE_BACKEND")
    #             pass
    #         else:
    #             backend = _backend_env_var

    if backend == "autograd":
        logger.debug("Setting autograd backend")
        _AUTOGRAD = True
        _delvar("_JAX")
        _delvar("_TORCH")
        _delvar("_SCIPY")
    elif backend == "scipy":
        logger.debug("Setting scipy backend")
        _SCIPY = True
        _delvar("_JAX")
        _delvar("_TORCH")
        _delvar("_AUTOGRAD")
    elif backend == "jax":
        logger.debug("Setting jax backend")
        _JAX = True
        _delvar("_SCIPY")
        _delvar("_TORCH")
        _delvar("_AUTOGRAD")
    elif backend == "torch":
        _TORCH = True
        logger.debug("Setting torch backend")
        _delvar("_SCIPY")
        _delvar("_JAX")
        _delvar("_AUTOGRAD")
    elif backend == "numpy":
        _NUMPY = True
        logger.debug("Setting numpy backend")
        _delvar("_SCIPY")
        _delvar("_JAX")
        _delvar("_AUTOGRAD")
        _delvar("_TORCH")
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Please choose between 'numpy', 'scipy', 'jax', 'torch' and 'autograd'."
        )
    _reload_package()


def _reload_package():

    import importlib
    import sys

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


def _grad(f):
    raise NotImplementedError(f"grad is not implemented for {BACKEND} backend.")


if "_SCIPY" in globals():

    import numpy

    grad = _grad
    backend = numpy
elif "_AUTOGRAD" in globals():
    from autograd import grad, numpy

    backend = numpy
elif "_JAX" in globals():
    if HAS_JAX:

        from jax.config import config

        config.update("jax_platform_name", "cpu")
        config.update("jax_enable_x64", True)

        # TODO: jax eig not implemented on GPU
        # see https://github.com/google/jax/issues/1259

        # TODO: support jax properly (is it faster than autograd? use jit?)
        # jax does not support eig
        # for autodif wrt eigenvectors yet.
        # see: https://github.com/google/jax/issues/2748

        # if DEVICE == "cpu":
        #     config.update("jax_platform_name", "cpu")
        # else:
        #     config.update("jax_platform_name", "gpu")
        # from jax import grad, numpy
        from jax import numpy

        grad = _grad
        backend = numpy

    else:
        logger.warning("jax not found. Falling back to default numpy backend.")
        set_backend("numpy")
elif "_TORCH" in globals():
    if HAS_TORCH:
        import numpy
        import torch

        # torch.set_default_tensor_type(torch.cuda.FloatTensor)

        backend = torch

        def _array(a, **kwargs):
            if isinstance(a, backend.Tensor):
                return a.to(torch.device(DEVICE))
            else:
                return backend.tensor(a, **kwargs).to(torch.device(DEVICE))

        backend.array = _array

        def grad(f):
            def df(x, *args, **kwargs):
                x = backend.array(x, dtype=bk.float64)
                _x = x.clone().detach().requires_grad_(True)
                out = backend.autograd.grad(
                    f(_x, *args, **kwargs), _x, allow_unused=True
                )[0]
                return out

            return df

    else:
        logger.warning("pytorch not found. Falling back to default numpy backend.")
        set_backend("numpy")
else:
    import numpy

    grad = _grad
    backend = numpy


def get_device():
    return "cuda" if "_GPU_DEVICE" in globals() else "cpu"


BACKEND = get_backend()
DEVICE = get_device()

_backend_env_var = os.environ.get("NANNOS_BACKEND")

if _backend_env_var in available_backends and _backend_env_var is not None:
    if BACKEND != _backend_env_var and "_FORCE_BACKEND" not in globals():
        logger.debug(f"Found environment variable NANNOS_BACKEND={_backend_env_var}")
        set_backend(_backend_env_var)

from . import optimize
from .constants import *
from .excitation import *
from .lattice import *
from .parallel import *
from .sample import *
from .simulation import *
from .utils import *
