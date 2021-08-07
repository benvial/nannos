#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

from . import numpy as np
from .constants import pi

__all__ = ["Lattice"]


class Lattice:
    def __init__(self, basis_vectors):
        self.basis_vectors = basis_vectors

    @property
    def area(self):
        v = self.basis_vectors
        return np.linalg.norm(np.cross(v[0], v[1]))

    @property
    def matrix(self):
        return np.array(self.basis_vectors).T

    @property
    def reciprocal(self):
        return 2 * pi * np.linalg.inv(self.matrix).T

    # wavevectors()

    def get_harmonics(self, nG, method="circular"):
        if not int(nG) == nG:
            raise ValueError("nG must be integer.")
        if method == "circular":
            return circular_truncation(nG, self.reciprocal)
        elif method == "parallelogram":
            return parallelogramic_truncation(nG, self.reciprocal)
        else:
            raise ValueError(
                f"Unknown truncation method '{method}', please choose between 'circular' or 'parallelogram'."
            )


def parallelogramic_truncation(nG, Lk):
    u = [np.linalg.norm(l) for l in Lk]
    udot = np.dot(Lk[0], Lk[1])

    NGroot = int(np.sqrt(nG))
    if np.mod(NGroot, 2) == 0:
        NGroot -= 1

    M = NGroot // 2

    xG = range(-M, NGroot - M)
    G = np.meshgrid(xG, xG, indexing="ij")
    G = [g.flatten() for g in G]

    # sorting
    Gl2 = G[0] ** 2 * u[0] ** 2 + G[1] ** 2 * u[0] ** 2 + 2 * G[0] * G[1] * udot
    jsort = np.argsort(Gl2)
    Gsorted = [g[jsort] for g in G]

    # final G
    nG = NGroot ** 2
    G = np.array(Gsorted, dtype=int)[:, :nG]

    return G, nG


def circular_truncation(nG, Lk):
    u = [np.linalg.norm(l) for l in Lk]
    udot = np.dot(Lk[0], Lk[1])
    ucross = Lk[0][0] * Lk[1][1] - Lk[0][1] * Lk[1][0]

    circ_area = nG * np.abs(ucross)
    circ_radius = np.sqrt(circ_area / pi) + u[0] + u[1]

    u_extent = [
        1 + int(circ_radius / (q * np.sqrt(1.0 - udot ** 2 / (u[0] * u[1]) ** 2)))
        for q in u
    ]

    xG, yG = [range(-q, q + 1) for q in u_extent]
    G = np.meshgrid(xG, yG, indexing="ij")
    G = [g.flatten() for g in G]

    # sorting
    Gl2 = G[0] ** 2 * u[0] ** 2 + G[1] ** 2 * u[0] ** 2 + 2 * G[0] * G[1] * udot
    jsort = np.argsort(Gl2)
    Gsorted = [g[jsort] for g in G]
    Gl2 = Gl2[jsort]

    # final G
    nGtmp = (2 * u_extent[0] + 1) * (2 * u_extent[1] + 1)
    if nG < nGtmp:
        nGtmp = nG
    # removing the part outside the cycle
    tol = 1e-10 * max(u[0] ** 2, u[1] ** 2)
    for i in range(nGtmp - 1, -1, -1):
        if np.abs(Gl2[i] - Gl2[i - 1]) > tol:
            break
    nG = i

    G = np.array(Gsorted, dtype=int)[:, :nG]

    return G, nG
