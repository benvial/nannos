#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import shapely.geometry as sg

import nannos as nn
import nannos.geometry as ng

plt.ion()
plt.close("all")

lattice = nn.Lattice(((1, 0), (1.4, 1.5)))
# lattice = nn.Lattice(((1,0), (0., 1.5)))
# lattice = nn.Lattice(((1, 0), (-0.5, 3 ** 0.5 / 2)))
lattice = nn.Lattice(((1, 0), (0.5, 3 ** 0.5 / 2)))
# lattice = nn.Lattice(((1, 0.3   ), (0.2, 2)))
# lattice = nn.Lattice(((1, 0), (0,1)))

bv = lattice.basis_vectors
bk = nn.backend


radius = 0.3  # 3**0.5/4

n2 = 8
Nx, Ny = 2 ** n2, 2 ** n2
grid = bk.ones((Nx, Ny), dtype=float) * 5
x0 = bk.linspace(0, 1.0, Nx)
y0 = bk.linspace(0, 1.0, Ny)


M = lattice.matrix
Minv = bk.linalg.inv(M)

#
# x1 = x0*M[0][0] + y0*M[0][1]
# y1 = x0*M[1][0] + y0*M[1][1]
# x, y = bk.meshgrid(x1, y1, indexing="ij")


x_, y_ = bk.meshgrid(x0, y0, indexing="ij")

x = x_ * M[0][0] + y_ * M[0][1]
y = x_ * M[1][0] + y_ * M[1][1]

#
#
# ring = sg.Polygon([(0.1, 0.1), (0.9, 0.9), (0.1, 0.5)])
#
# mask = ng.shape_mask(ring, x, y)
# xsx


radius_x, radius_y = radius / 2, radius

xc, yc = 0.5, 0.5
xc1 = xc * M[0][0] + yc * M[0][1]
yc1 = xc * M[1][0] + yc * M[1][1]
hole = (x - xc1) ** 2 / radius_x ** 2 + (y - yc1) ** 2 / radius_y ** 2 < 1


R = ((x - xc1) ** 2 + (y - yc1) ** 2) ** 0.5
T = bk.arctan((y - yc1) / (x - xc1))

hole = R < 0.2 + 0.1 * bk.cos(T) - 0.00 * bk.sin(T) + 0.1 * bk.cos(
    2 * T
) + 0.02 * bk.sin(2 * T)

line = 0.2
nhs = 7
import random

for i in range(nhs):
    line += 0.3 * random.random() / nhs * bk.cos(i * T)

hole = R < line

grid[hole] = 1
plt.pcolormesh(
    x0,
    y0,
    grid.T,
    cmap="tab20c",
)
plt.axis("equal")

#
# radius=0.1
#
# xc, yc = 0.0, 0.0
# xc1 = xc * M[0][0] + yc * M[0][1]
# yc1 = xc * M[1][0] + yc * M[1][1]
# hole = (x - xc1) ** 2 + (y - yc1) ** 2 < radius ** 2
# grid[hole] = 2
#
#
# xc, yc = 1, 1
# xc1 = xc * M[0][0] + yc * M[0][1]
# yc1 = xc * M[1][0] + yc * M[1][1]
# hole = (x - xc1) ** 2 + (y - yc1) ** 2 < radius ** 2
# grid[hole] = 2
#
#
# xc, yc = 0, 1
# xc1 = xc * M[0][0] + yc * M[0][1]
# yc1 = xc * M[1][0] + yc * M[1][1]
# hole = (x - xc1) ** 2 + (y - yc1) ** 2 < radius ** 2
# grid[hole] = 2
#
#
# xc, yc = 1, 0
# xc1 = xc * M[0][0] + yc * M[0][1]
# yc1 = xc * M[1][0] + yc * M[1][1]
# hole = (x - xc1) ** 2 + (y - yc1) ** 2 < radius ** 2
# grid[hole] = 2


lx, ly = [bk.linalg.norm(v) for v in lattice.basis_vectors]

fig, ax = plt.subplots()

nperx, npery = 5, 5
for i in range(nperx):
    for j in range(npery):

        # im = ax.imshow(
        #     grid.T,
        #     origin="lower",
        #     extent=(0, 1, 0, 1),
        #     cmap="tab20c",
        #     interpolation="nearest",
        # )
        im = ax.pcolormesh(
            x0,
            y0,
            grid.T,
            cmap="tab20c",
        )
        matrix = bk.eye(3)
        matrix[:2, :2] = lattice.matrix
        transform = (
            mtransforms.Affine2D(matrix=matrix)
            .translate(i * bv[0][0], i * bv[0][1])
            .translate(j * bv[1][0], j * bv[1][1])
        )
        trans_data = transform + ax.transData
        im.set_transform(trans_data)
        # im.axes.clear()
        # ax.plot(*bv[0],"o")
        # ax.plot(*bv[1],"o")

ax.set_xlim(0, nperx * bv[0][0] + npery * bv[1][0])
ax.set_ylim(0, nperx * bv[0][1] + npery * bv[1][1])
# plt.axis("scaled")
ax.set_aspect("equal")


# ax.set_xlim(0, 5)
# ax.set_ylim(0, 5)
#
# fig, ax = plt.subplots()
# im = ax.pcolormesh(x0,y0,
#     grid.T,
#     cmap="tab20c",
# )
# ax.set_aspect("equal")
