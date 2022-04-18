#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

import matplotlib.pyplot as plt
import numpy as np

import nannos as nn

plt.ion()

plt.close("all")
bk = nn.backend
formulation = "original"

lattice = nn.Lattice([[1.0, 0], [0, 1.0]], discretization=2**11)
lattice = nn.Lattice(1.0, discretization=2**9)
sup = lattice.Layer("Superstrate", epsilon=1)
sub = lattice.Layer("Substrate", epsilon=1)
epsilon = lattice.ones() * 3
# hole = lattice.circle(center=(0.5, 0.5), radius=0.2)

hole = lattice.stripe(0.5, 0.1)
epsilon[hole] = 1
ms = lattice.Layer("Metasurface", thickness=0.5, epsilon=epsilon)
pw = nn.PlaneWave(wavelength=1.2, angles=(0, 0, 45))
nh = 211
sim = nn.Simulation([sup, ms, sub], pw, nh=nh, formulation=formulation)
R, T = sim.diffraction_efficiencies()
print(R, T, R + T)


#
#
M = ms.matrix
# M = ms.Qeps @ ms.matrix
test = np.abs(np.conj(M).T - M)
print(np.all(test < 1e-12))
np.all(np.abs(np.conj(M).T - M) < 1e-12)
#
# fig,ax=plt.subplots(1,2)
#
# re = ax[0].imshow(M.real)
# plt.colorbar(re,ax=ax[0])
# ax[0].set_axis_off()
#
# im = ax[1].imshow(M.imag)
# plt.colorbar(im,ax=ax[1])
# ax[1].set_axis_off()
# plt.tight_layout()
#
#
#
# plt.figure()
# plt.imshow(np.log10(test))
# plt.colorbar()
#
#
# import scipy
#
# q = np.linalg.eig(ms.matrix)
# q = scipy.linalg.eigh(ms.Qeps @ ms.matrix, ms.Qeps)
