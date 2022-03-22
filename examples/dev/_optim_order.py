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

plt.close("all")
plt.ion()

np.random.seed(1984)


#############################################################################
# Set a backend supporting automatic differentiation

# nn.set_backend("autograd")
nn.set_backend("torch")
nn.use_gpu(True)
no = nn.optimize

bk = nn.backend
formulation = "original"
nh = 100
L1 = [1.087, 0]
L2 = [0, 0.525]
rat_unit_cel = L1[0] / L2[1]
theta = 0.0 * bk.pi / 180
phi = 0.0 * bk.pi / 180
psi = 0.0 * bk.pi / 180

Nx = 2**7
Ny = 2**7

eps_sup = 1.45**2
eps_sub = 1.0
eps_min = 1.0
eps_max = 3.60**2

h_ms = 0.6

rfilt = Nx / 50
maxiter = 20
order_target = (1, 0)
freq_target = 1 / 1.050


def run(density, proj_level=None, rfilt=0, freq=1, nh=nh, psi=0):
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
    sim_TE = run(density, proj_level, rfilt, freq=freq_target, psi=0)
    _, T_TE = sim_TE.diffraction_efficiencies(orders=True)

    sim_TM = run(density, proj_level, rfilt, freq=freq_target, psi=nn.pi / 2)
    _, T_TM = sim_TM.diffraction_efficiencies(orders=True)
    return -0.5 * (
        sim_TE.get_order(T_TE, order_target) + sim_TM.get_order(T_TM, order_target)
    )


##############################################################################
# Define initial density

density0 = np.random.rand(Nx, Ny)
density0 = bk.array(density0)
density0 = 0.5 * (density0 + bk.fliplr(density0))
density0 = 0.5 * (density0 + bk.flipud(density0))
density0 = 0.5 * (density0 + density0.T)
density0 = no.apply_filter(density0, rfilt)
density0 = (density0 - density0.min()) / (density0.max() - density0.min())
density0 = density0.flatten()
density_plot0 = bk.reshape(density0, (Nx, Ny))


def imshow(s, *args, **kwargs):
    extent = (0, rat_unit_cel, 0, 1)
    if nn.DEVICE == "cuda":
        plt.imshow(s.T.cpu(), *args, extent=extent, **kwargs)
    else:
        plt.imshow(s.T, *args, extent=extent, **kwargs)


plt.figure()
imshow(density_plot0)
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
    imshow(density_fp)
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
    method="nlopt",
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

for psi in [0, nn.pi / 2]:
    sim = run(density_bin, None, 0, freq=freq_target, psi=psi)
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
imshow(density_bin)
plt.axis("off")
plt.colorbar()
plt.title(f"objective = {Ttarget}")
plt.tight_layout()
plt.show()
plt.pause(0.1)
