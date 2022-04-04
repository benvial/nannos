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
from matplotlib import colors

import nannos as nn

lattice = nn.Lattice([[1.0, 0], [0, 1.0]], discretization=(2**9, 2**9))
sup = lattice.Layer("Superstrate", epsilon=1)
ms = lattice.Layer("Metasurface", thickness=0.5)
sub = lattice.Layer("Substrate", epsilon=1)
ms.epsilon = lattice.ones() * 12.0
circ = lattice.circle(center=(0.5, 0.5), radius=0.2)
ms.epsilon[circ] = 1
pw = nn.PlaneWave(frequency=1.4, angles=(0, 0, 0))
stack = [sup, ms, sub]
sim = nn.Simulation(stack, pw, nh=51)
R, T = sim.diffraction_efficiencies()
M = ms.matrix[: sim.nh, : sim.nh]


cols = [
    "#f0f0f0",
    "#dba234",
    "#e2e2e2",
    "#b1a9a9",
    "#535e65",
    "#0a2534",
    "#1672a7",
]
cmap = colors.ListedColormap(cols)

x = range(sim.nh)
x1, x2 = np.meshgrid(range(sim.nh), range(sim.nh), indexing="ij")
y = np.log10(np.abs(M))
y = np.fliplr(y)
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.pcolormesh(x1, x2, y.T, cmap=cmap, ec="white", lw=0.4)
plt.axis("off")
plt.axis("equal")
plt.savefig(
    "bg.svg",
    bbox_inches="tight",
    pad_inches=0,
)
