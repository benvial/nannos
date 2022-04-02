#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from . import backend as bk


def plot_line(ax, point1, point2):
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    ax.plot(x_values, y_values, "k", lw=0.5)


def plot_unit_cell(ax, bv):
    point1 = [0, 0]
    point2 = [bv[0][0], bv[0][1]]
    plot_line(ax, point1, point2)
    point3 = [bv[0][0] + bv[1][0], bv[0][1] + bv[1][1]]
    plot_line(ax, point2, point3)
    point4 = [bv[1][0], bv[1][1]]
    plot_line(ax, point1, point4)
    plot_line(ax, point4, point3)


def plot_layer(
    lattice, grid, epsilon, nper=1, ax=None, cmap="tab20c", show_cell=False, **kwargs
):
    ax = ax or plt.gca()
    if isinstance(nper, int):
        nperx, npery = nper, nper
    elif hasattr(nper, "__len__") and len(nper) == 2:
        nperx, npery = nper
    else:
        raise ValueError(f"Wrong type for nper {nper}")

    bv = lattice.basis_vectors
    ims = []
    for i in range(nperx):
        for j in range(npery):
            im = ax.pcolormesh(
                grid[0],
                grid[1],
                epsilon,
                cmap=cmap,
                **kwargs,
            )
            # matrix = bk.eye(3)
            # matrix[:2, :2] = lattice.matrix
            # mtransforms.Affine2D(matrix=matrix)
            transform = (
                mtransforms.Affine2D()
                .translate(i * bv[0][0], i * bv[0][1])
                .translate(j * bv[1][0], j * bv[1][1])
            )
            trans_data = transform + ax.transData
            im.set_transform(trans_data)
            ims.append(im)
    lx, ly = [bk.linalg.norm(v) for v in lattice.basis_vectors]
    l = max(lx, ly)
    delta = 0.1 * l
    ax.set_xlim(-delta, nperx * bv[0][0] + npery * bv[1][0] + delta)
    ax.set_ylim(-delta, nperx * bv[0][1] + npery * bv[1][1] + delta)
    if show_cell:
        plot_unit_cell(ax, bv)
    ax.set_aspect("equal")
    return ims
