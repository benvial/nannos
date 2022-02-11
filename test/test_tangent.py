#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


import pytest


#
def test_tangent():
    import numpy as npo

    import nannos as nn

    # nn.set_backend("torch")
    # nn.set_backend("numpy")
    bk = nn.backend

    n2 = 9
    Nx, Ny = 2 ** n2, 2 ** n2
    radius = 0.2
    grid = bk.ones((Nx, Ny), dtype=float)
    x0 = bk.linspace(0, 1.0, Nx)
    y0 = bk.linspace(0, 1.0, Ny)
    x, y = bk.meshgrid(x0, y0, indexing="xy")
    hole = (x - 0.3) ** 2 + (y - 0.3) ** 2 < radius ** 2
    # square = bk.logical_or(bk.abs(x + 0.7) < 0.2, bk.abs(y + 0.7) < 0.9)
    square0 = bk.logical_and(x > 0.7, x < 0.9)
    square1 = bk.logical_and(y > 0.2, y < 0.8)
    square = bk.logical_and(square1, square0)

    grid[hole] = 0
    grid[square] = 0

    t = nn.formulations.tangent.get_tangent_field(grid, normalize=True)
    ta = [npo.array(t[i]) for i in range(2)]
    normt = npo.linalg.norm(npo.array(ta), axis=0)
    assert npo.allclose(normt, 1)


#
#
#
#
# import matplotlib.pyplot as plt
#
# plt.ion()
# plt.close("all")
#
# plt.imshow(grid,cmap="tab20c")
# # plt.quiver(*ta,scale=100)
# dsp=10
# plt.quiver(x[::dsp,::dsp]*Nx,y[::dsp,::dsp]*Ny,ta[0][::dsp,::dsp],ta[1][::dsp,::dsp],scale=44)
# plt.axis("off")
# #
