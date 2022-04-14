#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


import matplotlib.pyplot as plt
import numpy as np
from pyinstrument import Profiler

import nannos as nn

plt.ion()
plt.close("all")

# nn.set_backend("torch")
nn.set_backend("numpy")

profiler = Profiler()
profiler.start()

nh = 111

lattice = nn.Lattice([[1.0, 0], [0, 1.0]], discretization=2**9)
sup = lattice.Layer("Superstrate", epsilon=1)
ms = lattice.Layer("Metasurface", thickness=0.5)
sub = lattice.Layer("Substrate", epsilon=1)
ms.epsilon = lattice.ones() * 12.0
circ = lattice.circle(center=(0.5, 0.5), radius=0.2)
ms.epsilon[circ] = 1
pw = nn.PlaneWave(wavelength=0.9, angles=(0, 0, 0))
stack = [sup, ms, sub]
sim = nn.Simulation(stack, pw, nh=nh)
R, T = sim.diffraction_efficiencies()

print(R, T, R + T)


profiler.stop()
profiler.open_in_browser()
# # profiler.print()
#
#
# def gram_schmidt(A):
#     """Orthogonalize a set of vectors stored as the columns of matrix A."""
#     # Get the number of vectors.
#     n = A.shape[1]
#     for j in range(n):
#         # To orthogonalize the vector in column j with respect to the
#         # previous vectors, subtract from it its projection onto
#         # each of the previous vectors.
#         for k in range(j):
#             A[:, j] -= np.dot(A[:, k].conj(), A[:, j]) * A[:, k]
#         A[:, j] = A[:, j] / np.linalg.norm(A[:, j])
#     return A
#
# # M = np.random.rand(3, 3)# + 1j*np.random.rand(3,3)
# # #
# # # M += M.T
# #
# M = sim.layers[1].matrix
# #
# w, phi = np.linalg.eig(M)
# #
# # phi_gs = gram_schmidt(phi)
# #
# out = np.conj(phi).T @ M @ phi
# # print(M)
# # print(np.abs(out))
#
layer = sim.layers[1]
Qeps = layer.Qeps
#
# phi = sim.layers[1].eigenvectors
# phi = gram_schmidt(phi)
#
# test = phi.T@Qeps @phi
matrix = sim.omega**2 * layer.Peps @ layer.Pmu - (
    layer.Peps @ layer.Keps + layer.Kmu @ layer.Pmu
)

Qeps = sim.omega**2 * np.eye(sim.nh * 2) - layer.Keps
matrix1 = layer.Peps @ layer.Qeps - layer.Kmu


M2 = Qeps @ layer.Peps @ Qeps - sim.omega**2 * layer.Kmu
N2 = Qeps
w, phi = np.linalg.eig(M2)
#
# import scipy
# w, phi = scipy.linalg.eig(M2,M2)
#
#
#
#
# w, phi = np.linalg.eig(matrix1)
#
# # phi = gram_schmidt(phi)
#
# out =np.conj(phi).T @ Qeps @ phi
#

#
# # out /= np.diag(out)
#
# plt.imshow(np.abs(out))
# plt.imshow(np.abs(out-np.diag(np.diag(out))))
# plt.colorbar()
#
#
#
# # # plt.imshow(np.abs(M2))
# # w, vl, vr = scipy.linalg.eig(matrix, left=True)
# # out = vl.T.conj()@matrix@vr


def _group_similar(items, comparer):
    """Combines similar items into groups.
    Args:
      items: The list of items to group.
      comparer: Determines if two items are similar.
    Returns:
      A list of groups of items.
    """

    groups = []
    used = set()
    for i in range(len(items)):
        if i not in used:
            group = [items[i]]
            for j in range(i + 1, len(items)):
                if j not in used and comparer(items[i], items[j]):
                    used.add(j)
                    group.append(items[j])
            groups.append(group)
    return groups


def orthogonal_eigendecompose(
    matrix,
    rtol=1e-12,
    atol=1e-12,
):
    """An eigendecomposition that ensures eigenvectors are orthogonal.
    numpy.linalg.eig doesn't guarantee that eigenvectors from the same
    eigenspace will be perpendicular. This method uses Gram-Schmidt to recover
    a perpendicular set. It further checks that all eigenvectors are
    perpendicular and raises an ArithmeticError otherwise.
    Args:
        matrix: The matrix to decompose.
        rtol: Relative threshold for determining whether eigenvalues are from
              the same eigenspace and whether eigenvectors are perpendicular.
        atol: Absolute threshold for determining whether eigenvalues are from
              the same eigenspace and whether eigenvectors are perpendicular.
    Returns:
        The eigenvalues and column eigenvectors. The i'th eigenvalue is
        associated with the i'th column eigenvector.
    Raises:
        ArithmeticError: Failed to find perpendicular eigenvectors.
    """
    vals, cols = np.linalg.eig(matrix)
    vecs = [cols[:, i] for i in range(len(cols))]

    # Convert list of row arrays to list of column arrays.
    for i in range(len(vecs)):
        vecs[i] = np.reshape(vecs[i], (len(vecs[i]), vecs[i].ndim))

    # Group by similar eigenvalue.
    n = len(vecs)
    groups = _group_similar(
        list(range(n)),
        lambda k1, k2: np.allclose(vals[k1], vals[k2], atol=atol, rtol=rtol),
    )

    # Remove overlap between eigenvectors with the same eigenvalue.
    for g in groups:
        q, _ = np.linalg.qr(np.hstack([vecs[i] for i in g]))
        for i in range(len(g)):
            vecs[g[i]] = q[:, i]

    return vals, np.array(vecs).T


w, phi = orthogonal_eigendecompose(matrix)

# phi /= (np.diag(phi)+1e-3)


out = np.conj(phi).T @ Qeps @ phi
n = np.diag(out)
phi = phi / n**0.5
out = np.conj(phi).T @ Qeps @ phi

# # out /= np.diag(out)
#
# plt.imshow(np.abs(out))
# plt.imshow(np.abs(out-np.diag(np.diag(out))))
plt.imshow(np.abs(out - np.eye(len(out))))
plt.colorbar()
