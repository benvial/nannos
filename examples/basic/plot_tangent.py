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
from nannos.utils import norm

plt.ion()

#############################################################################
# We will generate a field tangent to the material interface

nh = 151

n2 = 9
Nx, Ny = 2**n2, 2**n2
lattice = nn.Lattice(([1, 0], [0, 1]), discretization=(Nx, Ny))

x, y = lattice.grid()
circ = lattice.circle((0.3, 0.3), 0.25)
rect = lattice.rectangle((0.7, 0.7), (0.2, 0.5))
grid = lattice.ones() * (3 + 0.01j)
grid[circ] = 1
grid[rect] = 1


st = lattice.Layer("pat", thickness=1)
st.epsilon = grid
lays = [lattice.Layer("sup"), st, lattice.Layer("sub")]
pw = nn.PlaneWave(1.2)
sim = nn.Simulation(lays, pw, nh)


t = get_tangent_field(grid, sim.harmonics, normalize=False, type="fft")
norm_t = norm(t)
maxi = norm_t.max()
t = [t[i] / maxi for i in range(2)]


plt.figure()
st.plot()
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
maxi = norm_t.max()
topt = [topt[i] / maxi for i in range(2)]

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


#############################################################################
# Check formulations


def run(freq, t, formulation):
    st = lattice.Layer("pat", thickness=0.3, tangent_field=t)
    st.epsilon = grid
    pw = nn.PlaneWave(freq)
    lays = [lattice.Layer("sup"), st, lattice.Layer("sub")]
    sim = nn.Simulation(lays, pw, nh=nh, formulation=formulation)
    R, T = sim.diffraction_efficiencies()
    return T


plt.figure()
freqs = nn.backend.linspace(0.92, 0.94, 30)

compute_transmission = lambda freqs: run(freqs, None, "original")
freqs_adapted, transmission = nn.adaptive_sampler(
    compute_transmission,
    freqs,
)
plt.plot(freqs_adapted, transmission, label="original")

plt.pause(0.1)
compute_transmission = lambda freqs: run(freqs, t, "tangent")
freqs_adapted, transmission = nn.adaptive_sampler(
    compute_transmission,
    freqs,
)
plt.plot(freqs_adapted, transmission, label="tangent fft")

plt.pause(0.1)
compute_transmission = lambda freqs: run(freqs, topt, "tangent")
freqs_adapted, transmission = nn.adaptive_sampler(
    compute_transmission,
    freqs,
)
plt.plot(freqs_adapted, transmission, label="tangent opt")

plt.pause(0.1)

plt.xlim(freqs[0], freqs[-1])
plt.xlabel(r"frequency")
plt.ylabel("Transmission")
plt.legend()
plt.tight_layout()
