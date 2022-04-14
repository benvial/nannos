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


nh = 90
bk = nn.backend
# formulation = "tangent"
formulation = "original"

wl = 635
P = 350
L = 190
W = 100

# eps_Si = (3.87870 + 0.019221j) ** 2
eps_Si = (3.87 + 0.02j) ** 2
# eps_Si = (4.2 + 1j*0.3)**2


eps_SiO2 = 1.4573**2

# thicknesses = [200]


def get_order_comp(self, A, order, comp):
    c = 0 if comp == "x" else self.nh
    return A[c + self.get_order_index(order)]


plt.clf()

thicknesses_maxim = np.linspace(200, 500, 40)

for ih, H in enumerate(thicknesses_maxim):
    txx = txx_maxim[ih]
    tyx = tyx_maxim[ih]
    txy = txy_maxim[ih]
    tyy = tyy_maxim[ih]
    Jmaxim = 0.5 * np.array(
        [
            [txx + tyy + 1j * (tyx - txy), txx - tyy - 1j * (txy + tyx)],
            [txx - tyy + 1j * (tyx + txy), txx + tyy - 1j * (txy - tyx)],
        ]
    )
    # CEmaxim = np.abs((Jmaxim[0,0] - Jmaxim[0,1])/2)**2

    CEmaxim = np.abs((tyy - txx) / 2) ** 2

    plt.plot(H, CEmaxim, "+b")
    plt.pause(0.01)

thicknesses = np.linspace(200, 500, 61)


for ih, H in enumerate(thicknesses):

    lattice = nn.Lattice([[P, 0], [0, P]], discretization=2**9)
    sup = lattice.Layer("Superstrate", epsilon=eps_SiO2)
    sub = lattice.Layer("Substrate", epsilon=1)
    epsilon = lattice.ones()
    metaatom = lattice.rectangle((0.5 * P, 0.5 * P), (W, L))
    epsilon[metaatom] = eps_Si
    ms = lattice.Layer("Metasurface", thickness=H, epsilon=epsilon)
    pwx = nn.PlaneWave(wavelength=1 / 1 / wl, angles=(0, 0, 0 * nn.pi / 2))
    simx = nn.Simulation([sup, ms, sub], pwx, nh=nh, formulation=formulation)
    pwy = nn.PlaneWave(wavelength=1 / 1 / wl, angles=(0, 0, 1 * nn.pi / 2))
    simy = nn.Simulation([sup, ms, sub], pwy, nh=nh, formulation=formulation)

    # axN, bx0 = simx._get_amplitudes("Substrate",translate=True)
    #
    # simx.get_S_matrix()
    # axN = simx.S @ simx.a0
    # txx = get_order_comp(simx, axN[0][0], (0, 0), "x")*norma
    # tyx = get_order_comp(simx, axN[0][0], (0, 0), "y")*norma
    #
    #
    # simy.get_S_matrix()
    # ayN = simy.S @ simy.a0
    # # ayN, by0 = simy._get_amplitudes("Substrate",translate=True)
    # txy = get_order_comp(simy, ayN[0][0], (0, 0), "x")*norma
    # tyy = get_order_comp(simy, ayN[0][0], (0, 0), "y")*norma
    Rx, Tx = simx.diffraction_efficiencies()
    print(Tx)

    Ry, Ty = simy.diffraction_efficiencies()
    print(Ty)

    nin = (simx.layers[0].epsilon * simx.layers[0].mu) ** 0.5
    nout = (simx.layers[-1].epsilon * simx.layers[-1].mu) ** 0.5
    norma_t = 1 / (nout * nin) ** 0.5

    axN = simx.get_field_fourier("Substrate")[0, 0, 0:2]
    txx = simx.get_order(axN[0], (0, 0)) / norma_t
    tyx = simx.get_order(axN[1], (0, 0)) / norma_t

    ayN = simy.get_field_fourier("Substrate")[0, 0, 0:2]
    txy = simy.get_order(ayN[0], (0, 0)) / norma_t
    tyy = simy.get_order(ayN[1], (0, 0)) / norma_t

    J = 0.5 * np.array(
        [
            [txx + tyy + 1j * (tyx - txy), txx - tyy - 1j * (txy + tyx)],
            [txx - tyy + 1j * (tyx + txy), txx + tyy - 1j * (txy - tyx)],
        ]
    )
    CE = np.abs((J[0, 0] - J[0, 1]) / 2) ** 2
    CE = np.abs((tyy - txx) / 2) ** 2

    plt.plot(H, CE, "or")
    # plt.plot(H,Tx,"or")
    # plt.plot(H,Ty,"sg")
    plt.pause(0.1)
