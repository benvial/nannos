#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


"""
Photonic crystal slab
=====================

Metasurface with holes.
"""

# sphinx_gallery_thumbnail_number = 1

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import nannos as nn

plt.close("all")
plt.ion()

##############################################################################
# We will study a benchmark of hole in a dielectric surface similar to
# those studied in :cite:p:`Fan2002`.

nh = 100
L1 = [1.0, 0]
L2 = [0, 1.0]
freq = 1.4
theta = 30.0 * np.pi / 180
phi = 0.0 * np.pi / 180
psi = 0.0 * np.pi / 180

Nx = 2 ** 9
Ny = 2 ** 9

eps_sup = 1.0
eps_pattern = 12.0
eps_hole = 1.0
eps_sub = 1.0
h = 0.5

radius = 0.2
epsgrid = np.ones((Nx, Ny), dtype=float) * eps_pattern
x0 = np.linspace(0, 1.0, Nx)
y0 = np.linspace(0, 1.0, Ny)
x, y = np.meshgrid(x0, y0, indexing="ij")
hole = (x - 0.5) ** 2 + (y - 0.5) ** 2 < radius ** 2
epsgrid[hole] = eps_hole
#
#
# ##############################################################################
# # Visualize the permittivity
#
# cmap = mpl.colors.ListedColormap(["#ffe7c2", "#232a4e"])
#
# bounds = [eps_hole, eps_pattern]
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
# plt.imshow(epsgrid, cmap=cmap, extent=(0, 1, 0, 1))
# plt.colorbar(ticks=bounds)
# plt.xlabel("$x$")
# plt.ylabel("$y$")
# plt.title(r"permittitivity $\varepsilon(x,y)$")
# plt.tight_layout()
# plt.show()
#

##############################################################################
# Define the lattice

lattice = nn.Lattice((L1, L2))

##############################################################################
# Define the layers

sup = nn.Layer("Superstrate", epsilon=eps_sup)
ms = nn.Layer("Metasurface", thickness=h)
sub = nn.Layer("Substrate", epsilon=eps_sub)


##############################################################################
# Define the pattern and add it to the metasurface layer

pattern = nn.Pattern(epsgrid, name="hole")
ms.add_pattern(pattern)

##############################################################################
# Define the incident plane wave

pw = nn.PlaneWave(frequency=freq, angles=(theta, phi, psi))

##############################################################################
# Define the simulation

stack = [sup, ms, sub]
sim = nn.Simulation(lattice, stack, pw, nh)


##############################################################################
# Compute diffraction efficiencies

R, T = sim.diffraction_efficiencies()

##############################################################################
# Compute diffraction efficiencies per order

Ri, Ti = sim.diffraction_efficiencies(orders=True)
nmax = 5
print("Ri = ", Ri[:nmax])
print("Ti = ", Ti[:nmax])
print("R = ", R)
print("T = ", T)
print("R+T = ", R + T)
#
#
# M = ms.matrix
#
# fig, ax = plt.subplots(1, 2)
# a = ax[0].imshow(M.real)
# plt.colorbar(a, ax=ax[0])
# b = ax[1].imshow(M.imag)
# plt.colorbar(b, ax=ax[1])
#
#
# test = (M - np.conj(M).T)# / np.mean(M)
#
# fig, ax = plt.subplots(1, 2)
#
# a = ax[0].imshow(test.real)
# plt.colorbar(a, ax=ax[0])
# b = ax[1].imshow(test.imag)
# plt.colorbar(b, ax=ax[1])
#
#
#
# w, v = np.linalg.eig(M)
# wh, vh = np.linalg.eigh(M)
#
#
# plt.figure()
# plt.plot(w.real,w.imag,"o")
# plt.plot(wh.real,wh.imag,"x")
#
#
# def check_symmetric(a, rtol=1e-03, atol=1e-01):
#     return np.allclose(a, a.T, rtol=rtol, atol=atol)
#
#
#
# AA = (M - np.conj(M))
# # AA /= np.linalg.norm(M)
# plt.imshow(AA.imag)
# plt.colorbar()
# MT = np.conj(M).T
# plt.figure()
# plt.imshow(M.imag)
# plt.colorbar()
# plt.figure()
# plt.imshow(MT.imag)
# plt.colorbar()
#
#
# np.allclose((M - np.conj(M).T) / np.linalg.norm(M), 0, rtol=1e-39, atol=1e-2)
