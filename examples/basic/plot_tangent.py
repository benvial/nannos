#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


"""
Tangent field
=============


"""


import matplotlib.pyplot as plt

import nannos as nn
from nannos.formulations.tangent import get_tangent_field

#############################################################################
# We will generate a field tangent to the material interface

nh = 151
lattice = nn.Lattice(([1, 0], [0, 1]), discretization=2**9)

x, y = lattice.grid()
circ = lattice.circle((0.3, 0.3), 0.25)
rect = lattice.rectangle((0.7, 0.7), (0.2, 0.5))
grid = lattice.ones() * (3 + 0.01j)
grid[circ] = 1
grid[rect] = 1

st = lattice.Layer("pat", thickness=1, epsilon=grid)
lays = [lattice.Layer("sup"), st, lattice.Layer("sub")]
pw = nn.PlaneWave(wavelength=1 / 1.2)
sim = nn.Simulation(lays, pw, nh)


#############################################################################
# FFT version:


t = get_tangent_field(grid, sim.harmonics, normalize=False, type="fft")

dsp = 10

plt.figure()
st.plot()
plt.quiver(
    x[::dsp, ::dsp],
    y[::dsp, ::dsp],
    t[0][::dsp, ::dsp],
    t[1][::dsp, ::dsp],
    scale=20,
)
plt.axis("scaled")
_ = plt.axis("off")
plt.show()

#############################################################################
# Optimized version

topt = get_tangent_field(grid, sim.harmonics, normalize=False, type="opt")

plt.figure()
st.plot()
plt.quiver(
    x[::dsp, ::dsp],
    y[::dsp, ::dsp],
    topt[0][::dsp, ::dsp],
    topt[1][::dsp, ::dsp],
    scale=20,
)
plt.axis("scaled")
_ = plt.axis("off")
plt.show()
