#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


__all__ = ["Layer", "Pattern"]

from copy import copy

from . import get_backend
from . import numpy as np
from .simulation import block


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

    def __init__(self, name="layer", thickness=0, epsilon=1, mu=1):
        if thickness is not None:
            if thickness < 0:
                raise ValueError("thickness must be positive.")
        self.name = name
        self.thickness = thickness
        self.epsilon = epsilon
        self.mu = mu
        self.iscopy = False
        self.original = self
        self.patterns = []

        if self.mu != 1:
            # TODO: case with magnetic materials
            raise NotImplementedError(
                "Permeability different than unity not yet implemented."
            )

    def __repr__(self):
        return f"Layer {self.name}"

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

        is_mu_anisotropic = np.shape(mu)[:2] == (3, 3)
        is_epsilon_anisotropic = np.shape(epsilon)[:2] == (3, 3)

        if is_mu_anisotropic or is_epsilon_anisotropic:
            # TODO: anisotropic uniform layer
            raise NotImplementedError("Uniform layer material must be isotropic")

        IdG = np.eye(2 * nh)

        I = np.eye(nh)
        if is_epsilon_anisotropic:
            _epsilon = block(
                [
                    [epsilon[0, 0] * I, epsilon[0, 1] * I],
                    [epsilon[1, 0] * I, epsilon[1, 1] * I],
                ]
            )
        else:
            _epsilon = epsilon

        if is_mu_anisotropic:
            _mu = block([[mu[0, 0] * I, mu[0, 1] * I], [mu[1, 0] * I, mu[1, 1] * I]])
        else:
            _mu = mu

        q = (
            np.array(
                _epsilon * _mu * omega ** 2 - kx ** 2 - ky ** 2,
                dtype=complex,
            )
            ** 0.5
        )
        q = np.where(np.imag(q) < 0.0, -q, q)
        self.eigenvalues = np.concatenate((q, q))
        self.eigenvectors = np.eye(2 * len(kx))
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
            # FIXME: This gets slow because of the implementation
            # workaround to cpmpute eigenvalues with jax
            # TODO: implement custom autodiff rules for the evp with jax
            if get_backend() == "jax":
                from ._jax_eig_workaround import eig_jax

                eig_func = eig_jax
            else:
                eig_func = np.linalg.eig

            # eig_func = np.linalg.eig

            w, v = eig_func(matrix)
            q = w ** 0.5
            q = np.where(np.imag(q) < 0.0, -q, q)
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

    def add_pattern(self, pattern):
        """Add a pattern to the layer.

        Parameters
        ----------
        pattern : :class:`~nannos.Pattern`
            The pattern defined as a 2d grid on the unit cell.

        """
        self.patterns.append(pattern)

    @property
    def is_uniform(self):
        """Check if layer is uniform.

        Returns
        -------
        bool
            ``True`` if the layer is uniform, ``False`` if not.

        """
        return len(self.patterns) == 0


class Pattern:
    """A pattern object.

    Parameters
    ----------
    epsilon : array_like
        Permittivity `epsilon` (the default is 1).
    mu : array_like
        Permeability `mu` (the default is 1).
    name : str
        Name of the pattern (the default is "pattern").
    grid : array_like
        A 2d grid on which the pattern is defined (the default is None).



    """

    def __init__(self, epsilon=1, mu=1, name="pattern", grid=None):
        self.name = name
        self.grid = grid
        self.epsilon = epsilon
        self.mu = mu


#
#
#
# def eig2by2(M):
#     tr = M[0, 0] + M[1,1]
#     det = M[0, 0] * M[1,1] - M[1, 0] + M[0,1]
#     l0 = 0.5 * (tr - (tr ** 2 - 4 * det ) ** 0.5)
#     l1 = 0.5 * (tr + (tr ** 2 - 4 * det ) ** 0.5)
#     v0 = [-M[0, 1], M[0, 0] - l0]
#     v1 = [M[1, 1] - l1, -M[1, 0]]
#
#     eigenvalues = np.array([l0, l1])
#     # eigenvectors = np.array([v0, v1])
#     if  M[0, 0] - l0 == 0 or  M[1, 1] - l1 ==0:
#         eigenvectors = np.eye(2)
#     else:
#         eigenvectors = np.array([np.array(v)/(v[0]**2 + v[1]**2)**0.5 for v in [v0, v1]]).T
#
#     return eigenvalues, eigenvectors
