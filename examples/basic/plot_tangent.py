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
import numpy as npo

import nannos as nn
from nannos import backend as bk
from nannos.formulations.tangent import get_tangent_field
from nannos.utils import norm

#############################################################################
# We will generate a field tangent to the material interface


n2 = 9
Nx, Ny = 2 ** n2, 2 ** n2
radius = 0.25
x0 = bk.linspace(0, 1.0, Nx)
y0 = bk.linspace(0, 1.0, Ny)
x, y = bk.meshgrid(x0, y0, indexing="ij")
hole = (x - 0.3) ** 2 + (y - 0.3) ** 2 < radius ** 2
square0 = bk.logical_and(x > 0.7, x < 0.9)
square1 = bk.logical_and(y > 0.2, y < 0.8)
square = bk.logical_and(square1, square0)
grid = bk.ones((Nx, Ny), dtype=float) * 5
grid[hole] = 1
grid[square] = 1


st = nn.Layer("pat", thickness=1)
st.add_pattern(nn.Pattern(grid))
lays = [nn.Layer("sup"), st, nn.Layer("sub")]
pw = nn.PlaneWave(1.2)
sim = nn.Simulation(nn.Lattice(((1, 0), (0, 1))), lays, pw, nh=100)


t = get_tangent_field(grid, sim.harmonics, normalize=False, type="fft")
norm_t = norm(t)
maxi = bk.max(norm_t)
t = [t[i] / maxi for i in range(2)]


plt.figure()
plt.imshow(grid.T, cmap="tab20c", origin="lower", extent=(0, 1, 0, 1))
dsp = 10
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
norm_t = norm(topt)
maxi = bk.max(norm_t)
topt = [topt[i] / maxi for i in range(2)]

plt.figure()
plt.imshow(grid.T, cmap="tab20c", origin="lower", extent=(0, 1, 0, 1))
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


#############################################################################
# Check formulations


def run(t):
    st = nn.Layer("pat", thickness=1, tangent_field=t)
    st.add_pattern(nn.Pattern(grid))
    pw = nn.PlaneWave(1.2)
    lays = [nn.Layer("sup"), st, nn.Layer("sub")]
    sim = nn.Simulation(
        nn.Lattice(((1, 0), (0, 1))), lays, pw, nh=151, formulation="tangent"
    )
    R, T = sim.diffraction_efficiencies()
    print(f"R = {R}")
    print(f"T = {T}")
    print(f"R + T = {R + T}")


run(t)
run(topt)
