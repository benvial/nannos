#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


__all__ = ["Layer"]


from copy import copy

from . import BACKEND, DEVICE
from . import backend as bk
from . import get_types, jit
from .formulations.jones import get_jones_field
from .formulations.tangent import get_tangent_field
from .plot import *
from .utils import block
from .utils.helpers import _reseter, is_anisotropic, is_uniform

FLOAT, COMPLEX = get_types()


class Layer:
    """A layer object.

    Parameters
    ----------
    name : str
        Name of the layer (the default is "layer").
    thickness : float
        Thickness of the layer (the default is 0). Must be positive.
    epsilon : complex
        Permittivity `epsilon` (the default is 1).
    mu : complex
        Permeability `mu` (the default is 1).


    """

    def __init__(
        self,
        name="layer",
        thickness=0,
        epsilon=1,
        mu=1,
        lattice=None,
        tangent_field=None,
        tangent_field_type="fft",
    ):
        if thickness is not None and thickness < 0:
            raise ValueError("thickness must be positive.")
        self.name = name
        self.thickness = thickness
        self.epsilon = bk.array(epsilon, dtype=COMPLEX)
        self.mu = bk.array(mu, dtype=COMPLEX)
        self.lattice = lattice
        self.tangent_field = tangent_field
        self.tangent_field_type = tangent_field_type
        self.iscopy = False
        self.original = self

        if self.mu != 1:
            # TODO: case with magnetic materials
            raise NotImplementedError(
                "Permeability different than unity not yet implemented."
            )

    def __repr__(self):
        return f"{self.name}"

    def plot(
        self,
        nper=1,
        ax=None,
        cmap="tab20c",
        show_cell=False,
        comp="re",
        cellstyle="w-",
        **kwargs,
    ):
        lattice = self.lattice
        if comp not in ["re", "im"]:
            raise ValueError(f"Unknown component {comp}. must be `re` or `im`.")
        toplot = self.epsilon.real if comp == "re" else self.epsilon.imag
        return plot_layer(
            lattice,
            lattice.grid,
            toplot,
            nper,
            ax,
            cmap,
            show_cell,
            cellstyle,
            **kwargs,
        )

    @property
    def is_solved(self):
        return hasattr(self, "eigenvalues") and hasattr(self, "eigenvectors")

    def reset(self, param="all"):
        if param == "eig":
            _reseter(self, "eigenvalues")
            _reseter(self, "eigenvectors")
        if param == "matrix":
            _reseter(self, "matrix")
        if param == "all":
            self.reset("eig")
            self.reset("matrix")

    def solve_uniform(self, omega, kx, ky, nh):
        """Solve for eigenmodes and eigenvalues of a uniform layer.

        Parameters
        ----------
        omega : float
            Pulsation.
        kx : array_like
            Transverse wavenumber x component.
        ky : array_like
            Transverse wavenumber y component.
        nh : int
            Number of harmonics.

        Returns
        -------
        tuple
            `(q,psi)`, with eigenvalues `q` and eigenvectors `psi`.

        """
        epsilon = self.epsilon
        mu = self.mu

        if self.is_mu_anisotropic or self.is_epsilon_anisotropic:
            # TODO: anisotropic uniform layer
            raise NotImplementedError("Uniform layer material must be isotropic")

        Id = bk.eye(nh)
        if self.is_epsilon_anisotropic:
            _epsilon = block(
                [
                    [epsilon[0, 0] * Id, epsilon[0, 1] * Id],
                    [epsilon[1, 0] * Id, epsilon[1, 1] * Id],
                ]
            )
        else:
            _epsilon = epsilon

        if self.is_mu_anisotropic:
            _mu = block(
                [[mu[0, 0] * Id, mu[0, 1] * Id], [mu[1, 0] * Id, mu[1, 1] * Id]]
            )
        else:
            _mu = mu

        q = (
            bk.array(
                _epsilon * _mu * omega**2 - kx**2 - ky**2,
                dtype=COMPLEX,
            )
            ** 0.5
        )
        # q = bk.where(bk.imag(q) < 0.0, -q, q)
        self.eigenvalues = bk.array(bk.hstack((q, q)))
        self.eigenvectors = bk.array(bk.eye(2 * len(kx), dtype=COMPLEX))
        return self.eigenvalues, self.eigenvectors

    def solve_eigenproblem(self, matrix):
        """Solve the eigenproblem for a patterned layer.

        Parameters
        ----------
        matrix : array_like
            The matrix which we search for eigenvalues and eigenvectors.

        Returns
        -------
        tuple
            `(q,psi)`, with eigenvalues `q` and eigenvectors `psi`.

        """
        if self.iscopy:
            self.eigenvalues, self.eigenvectors = (
                self.original.eigenvalues,
                self.original.eigenvectors,
            )

        else:
            # TODO: implement custom autodiff rules for the evp with jax
            # if get_backend() == "jax" and get_device()=="cuda":
            if BACKEND == "jax" and DEVICE == "cuda":
                # fix for GPU
                # from ._jax_eig_workaround import eig_jax
                import jax

                w, v = jax.jit(jax.numpy.linalg.eig, backend="cpu")(matrix)
                w = jax.device_put(w, jax.devices("gpu")[0])
                v = jax.device_put(v, jax.devices("gpu")[0])

            else:
                eig_func = bk.linalg.eig
                eig_func = jit(bk.linalg.eig)
                w, v = eig_func(matrix)
            q = w**0.5
            q = bk.where(bk.imag(q) < 0.0, -q, q)
            self.eigenvalues, self.eigenvectors = q, v
        return self.eigenvalues, self.eigenvectors

    def copy(self, name=None):
        """Copy a layer.

        Returns
        -------
        nannos.Layer
            A copy of the layer.

        """
        cp = copy(self)
        cp.iscopy = True
        cp.original = self
        if name is None:
            cp.name += " (copy)"
        else:
            cp.name = name
        return cp

    @property
    def is_epsilon_anisotropic(self):
        return is_anisotropic(self.epsilon)

    @property
    def is_mu_anisotropic(self):
        return is_anisotropic(self.mu)

    @property
    def is_uniform(self):
        """Check if layer is uniform.

        Returns
        -------
        bool
            ``True`` if the layer is uniform, ``False`` if not.

        """
        return is_uniform(self.epsilon) and is_uniform(self.mu)

    @property
    def is_mu_uniform(self):
        return is_uniform(self.mu)

    @property
    def is_epsilon_uniform(self):
        return is_uniform(self.epsilon)

    def get_tangent_field(self, harmonics, normalize=False, type=None):
        if self.is_uniform:
            return None
        if self.tangent_field is not None:
            return self.tangent_field
        epsilon = self.epsilon
        epsilon_zz = epsilon[2, 2] if self.is_epsilon_anisotropic else epsilon
        type = type or self.tangent_field_type
        return get_tangent_field(
            epsilon_zz,
            harmonics,
            normalize=normalize,
            type=self.tangent_field_type,
        )

    def get_jones_field(self, t):
        return None if self.is_uniform else get_jones_field(t)


def _get_layer(id, layers, layer_names):
    """Helper to get layer index and name.

    Parameters
    ----------
    id : int or str
        The index of the layer or its name.
    layers : list of Layers objects
        The layer list (stack).
    id : list of str
        Names of the layers.

    Returns
    -------
    layer : nannos.Layer
        The layer object.
    index : nannos.Layer
        The layer index in the stack.
    """
    if isinstance(id, str):
        if id not in layer_names:
            raise ValueError(f"Unknown layer name {id}")
        for i, l in enumerate(layers):
            if l.name == id:
                return l, i
    elif isinstance(id, int):
        return layers[id], id
    elif hasattr(id, "name"):
        return _get_layer(id.name, layers, layer_names)
    else:
        raise ValueError(
            f"Wrong id for layer: {id}. Please use an integrer specifying the layer index or a string for the layer name."
        )
