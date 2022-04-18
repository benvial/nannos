#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

#
# Simulation of conversion efficiency of a geometric metasurface. (a) Unit
# cell configuration consisting of a rectangular Si structure on a SiO2 substrate.
# Circularly-polarized light of wavelength 635 nm is normally incident from the
# substrate to structure. P: 350 nm, L: 190 nm, W: 100 nm. The structure height
# H varies from 200 nm to 500 nm. (b) Comparison of calculated polarization
# conversion efficiency. Truncation orders are set to 10 for both directions.

import matplotlib.pyplot as plt
import numpy as np

import nannos as nn

outx = np.loadtxt("x.txt", skiprows=0, delimiter=",", dtype=str)
head = outx[0]
txx_maxim = np.array((outx[1:][:, head == "T0_Ex"]), dtype=complex).ravel()
tyx_maxim = np.array((outx[1:][:, head == "T0_Ey"]), dtype=complex).ravel()
outy = np.loadtxt("y.txt", skiprows=0, delimiter=",", dtype=str)
head = outy[0]
txy_maxim = np.array((outy[1:][:, head == "T0_Ex"]), dtype=complex).ravel()
tyy_maxim = np.array((outy[1:][:, head == "T0_Ey"]), dtype=complex).ravel()


nh = 60
bk = nn.backend
# formulation = "tangent"
formulation = "original"

wl = 635
P = 350
L = 190
W = 100

eps_Si = (3.87 + 1 * 0.02j) ** 2
eps_SiO2 = 1.4573**2

plt.clf()

# thicknesses_maxim = np.linspace(200, 500, 40)
#
# for ih, H in enumerate(thicknesses_maxim):
#     txx = txx_maxim[ih]
#     tyx = tyx_maxim[ih]
#     txy = txy_maxim[ih]
#     tyy = tyy_maxim[ih]
#     CEmaxim = np.abs((tyy - txx) / 2) ** 2
#     plt.plot(H, CEmaxim, "+b")
#     plt.pause(0.01)


def simu(H, angles):
    lattice = nn.Lattice([[P, 0], [0, P]], discretization=2**9)
    sup = lattice.Layer("Superstrate", epsilon=eps_SiO2)
    sub = lattice.Layer("Substrate", epsilon=1)
    epsilon = lattice.ones()
    metaatom = lattice.rectangle((0.5 * P, 0.5 * P), (W, L))
    epsilon[metaatom] = eps_Si
    ms = lattice.Layer("Metasurface", thickness=H, epsilon=epsilon)
    pw = nn.PlaneWave(wavelength=wl, angles=angles)
    return nn.Simulation([sup, ms, sub], pw, nh=nh, formulation=formulation)


nb_thick = 60
thicknesses = np.linspace(200, 500, nb_thick)
# thicknesses = [400]  #
conv_effs = np.zeros(nb_thick)
for ih, H in enumerate(thicknesses):
    if ih == 0:
        simx = simu(H, (0, 0, 0))
    else:
        simx.layers[1].thickness = H
        simx.get_S_matrix()
    # Rx, Tx = simx.diffraction_efficiencies()
    # print(Rx, Tx, Rx + Tx)
    # # print(Ry, Ty, Ry + Ty)
    # r1i, t1i = get_complex_orders(simx)
    # T1i = np.sum(np.abs(t1i) ** 2,axis=0)
    # R1i = np.sum(np.abs(r1i) ** 2,axis=0)
    # Rxi, Txi = simx.diffraction_efficiencies(orders=True)
    # assert np.allclose(R1i,Rxi)
    # assert np.allclose(T1i,Txi)
    # R1 = np.sum(R1i)
    # T1 = np.sum(T1i)
    # print(R1, T1, R1 + T1)
    # assert np.allclose(R1,Rx)
    # assert np.allclose(T1,Tx)

    rxi, txi = simx.diffraction_efficiencies(orders=True, complex=True)
    txx = simx.get_order(txi[0], (0, 0))

    if ih == 0:
        simy = simu(H, (0, 0, 90))
    else:
        simy.layers[1].thickness = H
        simy.get_S_matrix()
    ryi, tyi = simy.diffraction_efficiencies(orders=True, complex=True)
    tyy = simy.get_order(tyi[1], (0, 0))
    CE = np.abs((tyy - txx) / 2) ** 2
    conv_effs[ih] = CE

    plt.plot(H, CE, "or")
    plt.pause(0.1)

#
# plt.clf()
# plt.plot(thicknesses, conv_effs)
# plt.xlabel("H (nm)")
# plt.ylabel("conversion efficiency")
# plt.tight_layout()
# plt.show()
#
# ms.plot()
# plt.show()
# p = simx.plot_structure(pbr=True, nper=(2, 2))
# p.show_axes()
# p.view_xy()
# p.show()
