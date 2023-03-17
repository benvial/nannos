#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


import numpy as np
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter

import nannos as nn

np.random.seed(1234)


lattice = nn.Lattice([[1.0, 0], [0, 1.0]], discretization=2**8)
sup = lattice.Layer("Superstrate", epsilon=1)

lays = [sup]


Nx, Ny = lattice.discretization
density0 = np.random.rand(Nx, Ny)
density0 = np.array(density0)
density0 = 0.5 * (density0 + np.fliplr(density0))
density0 = 0.5 * (density0 + np.flipud(density0))
density0 = 0.5 * (density0 + density0.T)
density0 = gaussian_filter(density0, sigma=Nx / 20)
density0 = (density0 - density0.min()) / (density0.max() - density0.min())
density0[density0 < 0.33] = 0
density0[np.logical_and(density0 >= 0.33, density0 < 0.66)] = 0.5
density0[density0 >= 0.66] = 1


epsgrid = 10 * density0 + 1
meta = lattice.Layer("meta", thickness=0.2)
meta.epsilon = epsgrid
lays.append(meta)


for il, (radius, thickness) in enumerate(zip([0.4, 0.2, 0.1], [0.3, 0.7, 0.1])):
    hole = lattice.circle(center=(0.5, 0.5), radius=radius)
    ids = lattice.ones()
    epsgrid = ids * (np.random.rand(1) * 10 + 2)
    epsgrid[hole] = 1
    st = lattice.Layer(f"Pattern Hole {il}", thickness=thickness)
    st.epsilon = epsgrid
    lays.append(st)
sub = lattice.Layer("Substrate", epsilon=4)
lays.append(sub)


pw = nn.PlaneWave(0.75)
sim = nn.Simulation(lays, pw)


# colors = ["#ed7559", "#4589b5", "#cad45f", "#7a6773", "#ed59da"]

sargs = dict(height=0.5, vertical=True, position_x=0.05, position_y=0.25)
p = sim.plot_structure(dz=0.5, null_thickness=1.5, scalar_bar_args=sargs)
