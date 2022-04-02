#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import matplotlib.pyplot as plt

import nannos as nn
from nannos.geometry import *
from nannos.plot import *

plt.ion()
plt.close("all")

bk = nn.backend

# lattice = nn.Lattice(((1, 0), (0.6, 1.4)))
# lattice = nn.Lattice(((1,0), (0., 1.5)))
# lattice = nn.Lattice(((1, 0), (-0.5, 3 ** 0.5 / 2)))
# lattice = nn.Lattice(((1, 0), (0.5, 3 ** 0.5 / 2)))
# lattice = nn.Lattice(((1, 0.3   ), (0.2, 2)))
# lattice = nn.Lattice(((1, 0), (0,1)))


lattices = [
    nn.Lattice(((1, 0.0), (0.0, 1))),
    nn.Lattice(((1, 0), (0.6, 1.4))),
    nn.Lattice(((1, 0.1), (0.2, 0.8))),
]
lattice = lattices[1]

radius = 0.3  # 3**0.5/4

n2 = 10
Nx, Ny = 2**n2, 2**n2
epsilon = bk.ones((Nx, Ny), dtype=float) * 3


# radii = radius, radius
#
# grid = lattice.unit_grid(Nx, Ny)
# grid_trans = lattice.transform(grid)
# center = lattice.transform([0.5, 0.5])
# hole = (grid_trans[0] - center[0]) ** 2 / radii[0] ** 2 + (
#     grid_trans[1] - center[1]
# ) ** 2 / radii[1] ** 2 < 1

#
# grid = lattice.unit_grid(Nx, Ny)
# grid_trans = lattice.transform(grid)
# center = bk.array([0.5, 0.5])
# hole = (grid[0] - center[0]) ** 2 / radii[0] ** 2 + (
#     grid[1] - center[1]
# ) ** 2 / radii[1] ** 2 < 1
#


#
# x,y = grid_trans
# xc,yc = center
#
#
# import random
# random.seed(9)
# R = ((x - xc) ** 2 + (y - yc) ** 2) ** 0.5
# T = bk.arctan2((y - yc) , (x - xc))
# line = 0.3
# nhs = 7
#
# for i in range(nhs):
#     line += 0.4 * random.random() / nhs * bk.sin(i * T)
#
# hole = R < line
# epsilon[hole] = 1
#
#
# fig, ax = plt.subplots()
# plt.pcolormesh(
#     grid_trans[0],
#     grid_trans[1],
#     epsilon,
#     cmap="tab20c",
# )
# plt.axis("equal")
#
#
#
# fig, ax = plt.subplots()
# ims = plot_layer(lattice, grid_trans, epsilon, nper=(3,2), ax=ax)
# plt.axis("off")
# plt.colorbar(ims[0])

#


class Layer:
    def __init__(self, lattice, b=1):
        self.lattice = lattice
        self.b = b


class Lattice:
    def __init__(self, a=2):
        self.a = a

    def Layer(self, *args, **kwargs):
        return Layer(self, *args, **kwargs)


vertices = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.6)]
for lattice in lattices:
    epsilon = bk.ones((Nx, Ny), dtype=float) * 3
    tri = polygon(vertices, lattice, Nx, Ny)
    epsilon[tri] = 1
    x, y = lattice.grid(Nx, Ny)
    plt.figure()
    plt.pcolormesh(x, y, epsilon, cmap="tab20c")
    plt.colorbar()
    plt.axis("scaled")
    plt.show()
    [plt.plot(*v, "ok") for v in vertices]


center = 0.5, 0.5
radius_x = 0.2
radius_y = 0.1
#
#
#
for lattice in lattices:
    epsilon = bk.ones((Nx, Ny), dtype=float) * 3
    # tri = circle(center,radius_x, lattice)
    tri = ellipse(center, radius_x, radius_y, lattice, Nx, Ny, rotate=bk.pi / 4)
    epsilon[tri] = 1
    x_, y_ = lattice.unit_grid(Nx, Ny)
    x, y = lattice.transform((x_, y_))
    plt.figure()
    plt.pcolormesh(x, y, epsilon, cmap="tab20c")
    plt.colorbar()
    plt.axis("scaled")
    plt.show()
    plt.plot(*center, "ok")


center = 0.5, 0.5
radius_x = 0.2
radius_y = 0.1
lattice = lattices[0]
# tri = circle(center,radius_x, lattice)
ell1 = ellipse(center, radius_x, radius_y, lattice, Nx, Ny, rotate=bk.pi / 4)
ell2 = ellipse((0.4, 0.6), radius_x, radius_y, lattice, Nx, Ny, rotate=-bk.pi / 4)


epsilon = bk.ones((Nx, Ny), dtype=float)
epsilon[ell1 | ell2] = 3
x_, y_ = lattice.unit_grid(Nx, Ny)
x, y = lattice.transform((x_, y_))
plt.figure()
plt.pcolormesh(x, y, epsilon, cmap="tab20c")
plt.colorbar()
plt.axis("scaled")
plt.show()
plt.plot(*center, "ok")
epsilon = bk.ones((Nx, Ny), dtype=float)
epsilon[ell1 & ell2] = 3
x_, y_ = lattice.unit_grid(Nx, Ny)
x, y = lattice.transform((x_, y_))
plt.figure()
plt.pcolormesh(x, y, epsilon, cmap="tab20c")
plt.colorbar()
plt.axis("scaled")
plt.show()
plt.plot(*center, "ok")
epsilon = bk.ones((Nx, Ny), dtype=float)
epsilon[ell1 ^ ell2] = 3
x_, y_ = lattice.unit_grid(Nx, Ny)
x, y = lattice.transform((x_, y_))
plt.figure()
plt.pcolormesh(x, y, epsilon, cmap="tab20c")
plt.colorbar()
plt.axis("scaled")
plt.show()
plt.plot(*center, "ok")
