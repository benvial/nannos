from .__about__ import __author__, __description__, __version__

try:
    _ADJOINT
    from autograd import numpy
except:
    _ADJOINT = False
    import numpy

from .constants import *
from .excitation import *
from .lattice import *
from .layers import *
from .log import *
from .simulation import *


def set_backend(backend):
    """Set the numerical backend.

    Parameters
    ----------
    backend : str
        Either ``numpy`` or ``autograd``.


    """

    import importlib
    import sys

    global _ADJOINT
    if backend == "autograd":
        _ADJOINT = True
        log.info("Setting autograd backend")
    elif backend == "numpy":
        try:
            del _ADJOINT
        except:
            pass
        log.info("Setting numpy backend")
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. PLesase choose between 'numpy' and 'autograd'."
        )

    import nannos

    importlib.reload(nannos)

    its = [s for s in sys.modules.items() if s[0].startswith("nannos")]
    for k, v in its:
        importlib.reload(v)


def get_backend():
    return "autograd" if _ADJOINT else "numpy"
