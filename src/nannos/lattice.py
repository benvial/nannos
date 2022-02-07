#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


from . import backend as bk
from .constants import pi

__all__ = ["Lattice"]


class Lattice:
    """A lattice object.

    Parameters
    ----------
    basis_vectors : tuple
        The lattice vectors :math:`((u_x,u_y),(v_x,v_y))`.


    """

    def __init__(self, basis_vectors):
        self.basis_vectors = basis_vectors

    @property
    def area(self):
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

    def get_harmonics(self, nh, method="circular"):
        """Short summary.

        Parameters
        ----------
        nh : int
            Number of harmonics.
        method : str
            The truncation method, available values are "circular" and "parallelogrammic"
            (the default is "circular").

        Returns
        -------
        type
            Description of returned object.

        """
        if not int(nh) == nh:
            raise ValueError("nh must be integer.")
        if method == "circular":
            return _circular_truncation(nh, self.reciprocal)
        elif method == "parallelogrammic":
            return _parallelogramic_truncation(nh, self.reciprocal)
        else:
            raise ValueError(
                f"Unknown truncation method '{method}', please choose between 'circular' or 'parallelogrammic'."
            )


def _parallelogramic_truncation(nh, Lk):
    u = bk.array([bk.linalg.norm(l) for l in Lk])
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
    u = bk.array([bk.linalg.norm(l) for l in Lk])
    udot = bk.dot(Lk[0], Lk[1])
    ucross = bk.array(Lk[0][0] * Lk[1][1] - Lk[0][1] * Lk[1][0])

    circ_area = nh * bk.abs(ucross)
    circ_radius = bk.sqrt(circ_area / pi) + u[0] + u[1]

    u_extent = bk.array(
        [
            1 + int(circ_radius / (q * bk.sqrt(1.0 - udot**2 / (u[0] * u[1]) ** 2)))
            for q in u
        ]
    )
    xG, yG = [bk.array(bk.arange(-q, q + 1)) for q in u_extent]
    G = bk.meshgrid(xG, yG, indexing="ij")
    G = [g.flatten() for g in G]

    # print(u[])get_device()

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
