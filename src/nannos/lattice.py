#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

__all__ = ["Lattice"]

from . import backend as bk
from . import get_backend
from .constants import pi
from .geometry import *
from .layers import Layer
from .utils import is_scalar


class Lattice:
    """A lattice object defining the unit cell.

    Parameters
    ----------
    basis_vectors : tuple
        The lattice vectors :math:`((u_x,u_y),(v_x,v_y))`.
        For mono-periodic gratings, specify the x-periodicity with a float `a`.

    discretization : int or tuple of length 2
        Spatial discretization of the lattice. If given an integer N, the
        discretization will be (N, N).

    truncation : str
        The truncation method, available values are "circular" and "parallelogrammic" (the default is "circular").
        This has no effect for mono-periodic gratings.

    """

    def __init__(
        self, basis_vectors, discretization=(2**8, 2**8), truncation="circular"
    ):
        if is_scalar(discretization):
            discretization = [discretization, discretization]
        else:
            discretization = list(discretization)

        if truncation not in ["circular", "parallelogrammic"]:
            raise ValueError(
                f"Unknown truncation method '{truncation}', please choose between 'circular' and 'parallelogrammic'."
            )

        self.is_1D = is_scalar(basis_vectors)
        if self.is_1D:
            self.truncation = "1D"
            self.basis_vectors = (basis_vectors, 0), (0, 1)
            self.discretization = (discretization[0], 1)
        else:
            self.truncation = truncation
            self.basis_vectors = basis_vectors
            self.discretization = tuple(discretization)

    @property
    def area(self):
        if self.is_1D:
            return self.basis_vectors[0][0]
        else:
            v = self.basis_vectors
            return bk.linalg.norm(bk.cross(v[0], v[1]))

    @property
    def matrix(self):
        """Basis matrix.

        Returns
        -------
        array like
            Matrix containing the lattice vectors (L1,L2).

        """
        return bk.array(self.basis_vectors, dtype=bk.float64).T

    @property
    def reciprocal(self):
        """Reciprocal matrix.

        Returns
        -------
        array like
            Matrix containing the lattice vectors (K1,K2) in reciprocal space.

        """
        return 2 * pi * bk.linalg.inv(self.matrix).T

    def get_harmonics(self, nh):
        """Get the harmonics with a given truncation method.

        Parameters
        ----------
        nh : int
            Number of harmonics.

        Returns
        -------
        G : list of tuple of integers of length 2
            Harmonics (i1, i2).
        nh : int
            The number of harmonics after truncation.

        """
        if not int(nh) == nh:
            raise ValueError("nh must be integer.")
        if self.truncation == "circular":
            return _circular_truncation(nh, self.reciprocal)
        elif self.truncation == "parallelogrammic":
            return _parallelogramic_truncation(nh, self.reciprocal)
        else:
            return _one_dim_truncation(nh)

    def no1d(func):
        def inner(self, *args, **kwargs):
            if self.is_1D:
                raise ValueError(
                    f"Cannot use method {func.__name__} for 1D gratings, please use stripe"
                )
            return func(self, *args, **kwargs)

        return inner

    @property
    def unit_grid(self):
        """Unit grid in cartesian coordinates.

        Returns
        -------
        array like
            The unit grid of size equal to the attribute `discretization`.

        """
        Nx, Ny = self.discretization
        x0 = bk.linspace(0, 1.0, Nx)
        y0 = bk.linspace(0, 1.0, Ny)
        x_, y_ = bk.meshgrid(x0, y0, indexing="ij")
        grid = bk.stack([x_, y_])
        return grid

    @property
    def grid(self):
        """Grid in lattice vectors basis.

        Returns
        -------
        array like
            The grid of size equal to the attribute `discretization`.

        """
        return self.transform(self.unit_grid)

    def transform(self, grid):
        """Transform from cartesian to lattice coordinates.

        Parameters
        ----------
        grid : tuple of array like
            Input grid, typically obtained by meshgrid.

        Returns
        -------
        array like
            Transformed grid in lattice vectors basis.

        """
        if get_backend() == "torch":
            return bk.tensordot(self.matrix, grid.double(), dims=([1], [0]))
        else:
            return bk.tensordot(self.matrix, grid, axes=(1, 0))

    def ones(self):
        """Return a new array filled with ones.

        Returns
        -------
        array like
            A uniform complex 2D array with shape ``self.discretization``.

        """
        return bk.ones(self.discretization, dtype=bk.complex128)

    def geometry_mask(self, geom):
        """Return a geametry boolean mask discretized on the lattice grid.

        Parameters
        ----------
        geom : Shapely object
            The geometry.

        Returns
        -------
        array of bool
            The shape mask.

        """
        return geometry_mask(geom, self, *self.discretization)

    @no1d
    def polygon(self, vertices):
        return polygon(vertices, self, *self.discretization)

    @no1d
    def circle(self, center, radius):
        return circle(center, radius, self, *self.discretization)

    @no1d
    def ellipse(self, center, radii, rotate=0):
        return ellipse(center, radii, self, *self.discretization, rotate=rotate)

    @no1d
    def square(self, center, width, rotate=0):
        return square(center, width, self, *self.discretization, rotate=rotate)

    @no1d
    def rectangle(self, center, widths, rotate=0):
        return rectangle(center, widths, self, *self.discretization, rotate=rotate)

    def stripe(self, center, width):
        return abs(self.grid[0] - center) <= width / 2

    def Layer(
        self,
        name="layer",
        thickness=0,
        epsilon=1,
        mu=1,
        lattice=None,
        tangent_field=None,
        tangent_field_type="fft",
    ):
        return Layer(
            name,
            thickness,
            epsilon,
            mu,
            self,
            tangent_field,
            tangent_field_type,
        )


def _one_dim_truncation(nh):
    n = int((nh - 1) / 2)
    G = [(0, 0)]
    for i in range(1, n + 1):
        G.append((i, 0))
        G.append((-i, 0))
    return bk.array(G).T, len(G)


def _parallelogramic_truncation(nh, Lk):
    u = bk.array([bk.linalg.norm(value) for value in Lk])
    udot = bk.dot(Lk[0], Lk[1])

    NGroot = int((nh) ** 0.5)
    if NGroot % 2 == 0:
        NGroot -= 1

    M = NGroot // 2

    xG = bk.array(bk.arange(-M, NGroot - M))
    G = bk.meshgrid(xG, xG, indexing="ij")
    G = [g.flatten() for g in G]

    Gl2 = G[0] ** 2 * u[0] ** 2 + G[1] ** 2 * u[0] ** 2 + 2 * G[0] * G[1] * udot
    jsort = bk.argsort(Gl2)
    Gsorted = [g[jsort] for g in G]

    nh = NGroot**2

    G = bk.vstack(Gsorted)[:, :nh]

    return G, nh


def _circular_truncation(nh, Lk):
    u = bk.array([bk.linalg.norm(value) for value in Lk])
    udot = bk.dot(Lk[0], Lk[1])
    ucross = bk.array(Lk[0][0] * Lk[1][1] - Lk[0][1] * Lk[1][0])

    circ_area = nh * bk.abs(ucross)
    circ_radius = bk.sqrt(circ_area / pi) + u[0] + u[1]

    _int = int if get_backend() == "torch" else bk.int32

    u_extent = bk.array(
        [
            1 + _int(circ_radius / (q * bk.sqrt(1.0 - udot**2 / (u[0] * u[1]) ** 2)))
            for q in u
        ]
    )
    xG, yG = [bk.array(bk.arange(-q, q + 1)) for q in u_extent]
    G = bk.meshgrid(xG, yG, indexing="ij")
    G = [g.flatten() for g in G]

    Gl2 = bk.array(
        G[0] ** 2 * u[0] ** 2 + G[1] ** 2 * u[0] ** 2 + 2 * G[0] * G[1] * udot
    )
    jsort = bk.argsort(Gl2)
    Gsorted = [g[jsort] for g in G]
    Gl2 = Gl2[jsort]

    nGtmp = (2 * u_extent[0] + 1) * (2 * u_extent[1] + 1)
    if nh < nGtmp:
        nGtmp = nh

    tol = 1e-10 * max(u[0] ** 2, u[1] ** 2)
    for i in bk.arange(nGtmp - 1, -1, -1):
        if bk.abs(Gl2[i] - Gl2[i - 1]) > tol:
            break
    nh = i

    G = bk.vstack(Gsorted)[:, :nh]

    return G, nh
