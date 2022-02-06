#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

from .__about__ import __author__, __description__, __version__
from .log import *

HAS_CUDA = False


def has_torch():
    try:
        import torch

        return True
    except ModuleNotFoundError:
        return False


HAS_TORCH = has_torch()


def set_backend(backend):
    """Set the numerical backend.

    Parameters
    ----------
    backend : str
        Either ``numpy``, ``autograd``, ``torch`` or ``jax``.


    """

    import importlib
    import sys

    global TORCH
    global JAX
    global AUTOGRAD
    if backend == "autograd":
        AUTOGRAD = True
        log.info("Setting autograd backend")
        try:
            del JAX
        except:
            pass
        try:
            del TORCH
        except:
            pass
    elif backend == "jax":
        JAX = True
        log.info("Setting jax backend")
        try:
            del AUTOGRAD
        except:
            pass
        try:
            del TORCH
        except:
            pass
    elif backend == "torch":
        TORCH = True
        log.info("Setting torch backend")

        try:
            del JAX
        except:
            pass
        try:
            del AUTOGRAD
        except:
            pass
    elif backend == "numpy":
        try:
            del JAX
        except:
            try:
                del AUTOGRAD
            except:
                try:
                    del TORCH
                except:
                    pass
        log.info("Setting numpy backend")
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Please choose between 'numpy', 'jax', 'torch' and 'autograd'."
        )

    import nannos

    importlib.reload(nannos)

    its = [s for s in sys.modules.items() if s[0].startswith("nannos")]
    for k, v in its:
        importlib.reload(v)


def get_backend():
    try:
        AUTOGRAD
        return "autograd"
    except:
        try:
            JAX
            return "jax"
        except:
            try:
                TORCH
                return "torch"
            except:
                return "numpy"


try:
    JAX
    from jax.config import config

    config.update("jax_enable_x64", True)
    from jax import grad, numpy

    backend = numpy

    # TODO: support jax since it is faster than autograd
    # jax does not support numpy.linalg.eig yet
    # for autodif wrt eigenvectors
    # see: https://github.com/google/jax/issues/2748
except:
    try:

        TORCH
        if HAS_TORCH:
            import numpy
            import torch

            backend = torch

            def _array(a, **kwargs):
                if isinstance(a, backend.Tensor):
                    return a
                else:
                    return backend.tensor(a, **kwargs)

            backend.array = _array

            def grad(f):
                def df(x):
                    _x = x.clone().detach().requires_grad_(True)
                    return backend.autograd.grad(f(_x), _x, allow_unused=True)[0]

                return df

            HAS_CUDA = torch.cuda.is_available()
        else:

            log.info("torch not found. Falling back to default backend")
            set_backend("numpy")

    except:
        try:
            AUTOGRAD
            from autograd import grad, numpy

            backend = numpy
        except:
            import numpy

            backend = numpy
            grad = None


BACKEND = get_backend()


from .constants import *
from .excitation import *
from .lattice import *
from .layers import *
from .parallel import *
from .sample import *
from .simulation import *
from .utils import *
