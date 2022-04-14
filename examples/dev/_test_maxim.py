#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

import matplotlib.pyplot as plt
import numpy as np

import nannos as nn

plt.ion()

nh = 51
bk = nn.backend
# formulation = "tangent"
formulation = "original"

lattice = nn.Lattice([[1.0, 0], [0, 1.0]], discretization=2**9)
sup = lattice.Layer("Superstrate", epsilon=1)
sub = lattice.Layer("Substrate", epsilon=1)
epsilon = lattice.ones() * 4
# hole = lattice.circle((0.5,0.5),0.3)
hole = lattice.ellipse((0.5, 0.5), (0.1, 0.3), rotate=30)
epsilon[hole] = 1
ms = lattice.Layer("Metasurface", thickness=0.5, epsilon=epsilon)
pw = nn.PlaneWave(wavelength=1 / 1.3, angles=(0, 0, 1 * nn.pi / 2))
sim = nn.Simulation([sup, ms, sub], pw, nh=nh, formulation=formulation)
t = nn.tic()
sim.solve()
nn.toc(t)

t = nn.tic()
sim.get_S_matrix()
nn.toc(t)

t = nn.tic()
R, T = sim.diffraction_efficiencies()
print(R, T)
nn.toc(t)
print("-----")
t = nn.tic()
print("solved: ", sim.is_solved)
sim.get_layer("Metasurface").thickness = 1
# sim.layers[1].thickness = 1  # same as previous line
del sim.S
print("solved: ", sim.is_solved)
Rupdate, Tupdate = sim.diffraction_efficiencies()
print("solved: ", sim.is_solved)
print(Rupdate, Tupdate)
nn.toc(t)
print("-----")

t = nn.tic()
ms = lattice.Layer("Metasurface", thickness=1, epsilon=epsilon)
sim = nn.Simulation([sup, ms, sub], pw, nh=nh, formulation=formulation)
Rnew, Tnew = sim.diffraction_efficiencies()
print(Rnew, Tnew)
nn.toc(t)
print("-----")

assert Rnew == Rupdate
assert Tnew == Tupdate


aN, b0 = sim._solve_ext()


R, T = sim.diffraction_efficiencies(orders=True)


def get_order_comp(self, A, order, comp):
    c = 0 if comp == "x" else self.nh
    return A[c + self.get_order_index(order)]


tx = get_order_comp(sim, aN, (0, 0), "x")
ty = get_order_comp(sim, aN, (0, 0), "y")
print(tx)
print(ty)


rx = get_order_comp(sim, b0, (0, 0), "x")
ry = get_order_comp(sim, b0, (0, 0), "y")
print(rx)
print(ry)

print(abs(rx) ** 2)
print(abs(tx) ** 2)
print(abs(ry) ** 2)
print(abs(ty) ** 2)
