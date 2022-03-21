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


# sphinx_gallery_thumbnail_number = -1

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import nannos as nn
import nannos.optimize as no

plt.close("all")
plt.ion()

np.random.seed(1984)


#############################################################################
# Set a backend supporting automatic differentiation

# nn.set_backend("autograd")
nn.set_backend("torch")

bk = nn.backend
formulation = "original"
nh = 22
L1 = [1.0, 0]
L2 = [0, 1.0]
theta = 0.0 * bk.pi / 180
phi = 0.0 * bk.pi / 180
psi = 0.0 * bk.pi / 180

Nx = 2 ** 7
Ny = 2 ** 7

eps_sup = 1.0
eps_sub = 3.0
eps_min = 1.0
eps_max = 6.0

h_ms = 1

maxiter = 20
rfilt = Nx / 25
order_target = (0, 1)
freq_target = 1.2


def run(density, proj_level=None, rfilt=0, freq=1, nh=nh):
    density = bk.reshape(density, (Nx, Ny))
    density_f = no.apply_filter(density, rfilt)
    density_fp = (
        no.project(density_f, proj_level) if proj_level is not None else density_f
    )
    epsgrid = no.simp(density_fp, eps_min, eps_max, p=1)
    lattice = nn.Lattice((L1, L2))
    pw = nn.PlaneWave(frequency=freq, angles=(theta, phi, psi))
    sup = nn.Layer("Superstrate", epsilon=eps_sup)
    sub = nn.Layer("Substrate", epsilon=eps_sub)
    ms = nn.Layer("Metasurface", epsilon=1, thickness=h_ms)
    pattern = nn.Pattern(epsgrid, name="design")
    ms.add_pattern(pattern)
    stack = [sup, ms, sub]
    sim = nn.Simulation(lattice, stack, pw, nh, formulation=formulation)
    return sim


##############################################################################
# Define objective function


def fun(density, proj_level, rfilt):
    sim = run(density, proj_level, rfilt, freq=freq_target)
    R, T = sim.diffraction_efficiencies(orders=True)
    return -sim.get_order(T, order_target)


##############################################################################
# Define initial density

density0 = np.random.rand(Nx, Ny)
density0 = bk.array(density0)
density0 = 0.5 * (density0 + bk.fliplr(density0))
density0 = 0.5 * (density0 + bk.flipud(density0))
density0 = 0.5 * (density0 + density0.T)
# density0 = no.apply_filter(density0, rfilt)
density0 = (density0 - density0.min()) / (density0.max() - density0.min())
density0 = density0.flatten()
density_plot0 = bk.reshape(density0, (Nx, Ny))

plt.figure()
plt.imshow(density_plot0)
plt.colorbar()
plt.axis("off")
plt.title("initial density")
plt.tight_layout()


##############################################################################
# Define calback function

it = 0


def callback(x, y, proj_level, rfilt):
    global it
    print(f"iteration {it}")
    density = bk.reshape(x, (Nx, Ny))
    density_f = no.apply_filter(density, rfilt)
    density_fp = no.project(density_f, proj_level)
    # plt.figure()
    plt.clf()
    plt.imshow(density_fp)
    plt.axis("off")
    plt.colorbar()
    plt.title(f"iteration {it}, objective = {y:.5f}")
    plt.tight_layout()
    plt.show()
    plt.pause(0.1)
    it += 1


##############################################################################
# Create `TopologyOptimizer` object

opt = no.TopologyOptimizer(
    fun,
    density0,
    method="scipy",
    threshold=(0, 8),
    maxiter=maxiter,
    stopval=-0.99,
    args=(1, rfilt),
    callback=callback,
    options={},
)

##############################################################################
# Run the optimization

density_opt, f_opt = opt.minimize()


##############################################################################
# Postprocess to get a binary design


density_opt = bk.reshape(bk.array(density_opt), (Nx, Ny))
proj_level = 2 ** (opt.threshold[-1] - 1)
density_optf = no.apply_filter(density_opt, rfilt)
density_optfp = no.project(density_optf, proj_level)
density_bin = bk.ones_like(density_optfp)
density_bin[density_optfp < 0.5] = 0


sim = run(density_bin, None, 0, freq=freq_target)
R, T = sim.diffraction_efficiencies(orders=True)
print("Σ R = ", float(sum(R)))
print("Σ T = ", float(sum(T)))
print("Σ R + T = ", float(sum(R + T)))
Ttarget = sim.get_order(T, order_target)
print("")
print(f"Target transmission in order {order_target}")
print(f"===================================")
print(f"T_{order_target} = ", float(Ttarget))

plt.clf()
plt.imshow(density_bin)
plt.axis("off")
plt.colorbar()
plt.title(f"objective = {Ttarget}")
plt.tight_layout()
plt.show()
plt.pause(0.1)
