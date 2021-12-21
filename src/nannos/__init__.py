#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

from .__about__ import __author__, __description__, __version__
from .log import *


def set_backend(backend):
    """Set the numerical backend.

    Parameters
    ----------
    backend : str
        Either ``numpy``, ``autograd`` or ``jax``.


    """

    import importlib
    import sys

    global JAX
    global AUTOGRAD
    if backend == "autograd":
        AUTOGRAD = True
        log.info("Setting autograd backend")
        try:
            del JAX
        except:
            pass
    elif backend == "jax":
        JAX = True
        log.info("Setting jax backend")
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
                pass
            pass
        log.info("Setting numpy backend")
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Please choose between 'numpy' 'jax' and 'autograd'."
        )

    import nannos

    importlib.reload(nannos)

    its = [s for s in sys.modules.items() if s[0].startswith("nannos")]
    for k, v in its:
        importlib.reload(v)


def get_backend():
    try:
        AUTOGRAD
    except:
        try:
            JAX
            return "jax"
        except:
            return "numpy"
    return "autograd"


try:
    JAX
    from jax.config import config

    config.update("jax_enable_x64", True)
    from jax import grad, numpy

    # TODO: support jax since it is faster than autograd
    # jax does not support numpy.linalg.eig yet
    # for autodif wrt eigenvectors
    # see: https://github.com/google/jax/issues/2748
except:
    try:
        AUTOGRAD
        from autograd import grad, numpy
    except:
        import numpy

        grad = None


from .constants import *
from .excitation import *
from .lattice import *
from .layers import *
from .parallel import *
from .sample import *
from .simulation import *
from .utils import *
