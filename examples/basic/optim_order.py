#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


"""
Topology optimization
=====================

Design of a metasurface with maximum transmission into a given order.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import nannos as nn
from nannos.optimize import TopologyOptimizer, filter, project

plt.close("all")
plt.ion()

##############################################################################
# We will study a benchmark of hole in a dielectric surface

nn.set_backend("autograd")

formulation = "tangent"
# formulation = "original"

nh = 100
L1 = [1.0, 0]
L2 = [0, 1.0]
freq = 1.1
theta = 0.0 * np.pi / 180
phi = 0.0 * np.pi / 180
psi = 0.0 * np.pi / 180

Nx = 2**7
Ny = 2**7

eps_sup = 1.0
eps_slab = 7.0
eps_sub = 3.0
eps_min = 1.0
eps_max = 4.0

h_slab = 1.0
h_ms = 0.5

rfilt = Nx / 25
order_target = (0, 1)


def run(density, proj_level=None, rfilt=0, freq=1, nh=nh):
    density = np.reshape(density, (Nx, Ny))
    density_f = filter(density, rfilt)
    density_fp = project(density_f, proj_level) if proj_level is not None else density_f
    epsgrid = (eps_max - eps_min) * density_fp + eps_min

    ##############################################################################
    # Define the lattice

    lattice = nn.Lattice((L1, L2))

    ##############################################################################
    # Define the incident plane wave

    pw = nn.PlaneWave(frequency=freq, angles=(theta, phi, psi))

    ##############################################################################
    # Define the layers

    sup = nn.Layer("Superstrate", epsilon=eps_sup)
    slab = nn.Layer("Slab", epsilon=eps_slab, thickness=h_slab)
    sub = nn.Layer("Substrate", epsilon=eps_sub)

    ##############################################################################
    # Define the pattern and add it to the metasurface layer
    ms = nn.Layer("Metasurface", epsilon=1, thickness=h_ms)
    pattern = nn.Pattern(epsgrid, name="design")
    ms.add_pattern(pattern)
    stack = [sup, ms, sub]

    ##############################################################################
    # Define the simulation

    sim = nn.Simulation(lattice, stack, pw, nh, formulation=formulation)

    return sim


##############################################################################
# Unpatterned

freq_target = 0.8


def fun(density, proj_level, rfilt):
    sim = run(density, proj_level, rfilt, freq=freq_target)
    R, T = sim.diffraction_efficiencies(orders=True)
    return -sim.get_order(T, order_target)


density0 = np.random.rand(Nx, Ny)
density0 = 0.5 * (density0 + np.fliplr(density0))
density0 = 0.5 * (density0 + np.flipud(density0))
density0 = 0.5 * (density0 + np.transpose(density0))
density0 = filter(density0, rfilt)
density0 = (density0 - density0.min()) / (density0.max() - density0.min())
density0 = density0.flatten()
density_plot0 = np.reshape(density0, (Nx, Ny))

plt.figure()
plt.imshow(density_plot0)


def callback(x, y, rfilt, proj_level):
    density = np.reshape(x, (Nx, Ny))
    density_f = filter(density, rfilt)
    density_fp = project(density_f, proj_level)

    plt.clf()
    plt.imshow(density_fp)
    plt.axis("off")
    plt.colorbar()
    plt.title(f"objective = {y}")
    plt.tight_layout()
    plt.show()
    plt.pause(0.1)


opt = TopologyOptimizer(
    fun,
    density0,
    method="nlopt",
    threshold=(0, 8),
    maxiter=20,
    stopval=-0.99,
    args=(rfilt, 1),
    callback=callback,
    options={},
)


density_opt, Ropt = opt.minimize()
density_opt = np.reshape(density_opt, (Nx, Ny))
proj_level = 2 ** (opt.threshold[-1] - 1)
density_optf = filter(density_opt, rfilt)
density_optfp = project(density_optf, proj_level)

density_bin = np.ones_like(density_optfp)
density_bin[density_optfp < 0.5] = 0


sim = run(density_bin, None, 0, freq=freq_target)
R, T = sim.diffraction_efficiencies()
print(R, T)

plt.clf()
plt.imshow(density_bin)
plt.axis("off")
plt.colorbar()
plt.title(f"objective = {R}")
plt.tight_layout()
plt.show()
plt.pause(0.1)


freqs_ms = np.linspace(0.9 * freq_target, 1.1 * freq_target, 100)
Rms = []

for freq in freqs_ms:
    sim = run(density=density_bin, proj_level=None, rfilt=0, freq=freq, nh=nh)
    R, T = sim.diffraction_efficiencies()
    print(R)
    Rms.append(R)


plt.figure()
plt.plot(freqs_ms, Rms, "b")
plt.pause(0.1)


sim = run(density=density_bin, proj_level=None, rfilt=0, freq=freq_target, nh=nh)
R, T = sim.diffraction_efficiencies()
print(R)
