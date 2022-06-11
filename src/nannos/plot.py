#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pyvista

from . import backend as bk

pyvista.set_jupyter_backend("pythreejs")
pyvista.set_plot_theme("document")
pyvista.global_theme.background = "white"
# pyvista.global_theme.window_size = [600, 400]
pyvista.global_theme.axes.show = True
# pyvista.global_theme.smooth_shading = True
# pyvista.global_theme.antialiasing = True
# pyvista.global_theme.axes.box = True


def plot_line(ax, point1, point2, cellstyle="k-"):
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    ax.plot(x_values, y_values, cellstyle, lw=0.5)


def plot_unit_cell(ax, bv, cellstyle="-k"):
    point1 = [0, 0]
    point2 = [bv[0][0], bv[0][1]]
    plot_line(ax, point1, point2, cellstyle)
    point3 = [bv[0][0] + bv[1][0], bv[0][1] + bv[1][1]]
    plot_line(ax, point2, point3, cellstyle)
    point4 = [bv[1][0], bv[1][1]]
    plot_line(ax, point1, point4, cellstyle)
    plot_line(ax, point4, point3, cellstyle)


def plot_layer(
    lattice,
    grid,
    epsilon,
    nper=1,
    ax=None,
    cmap="tab20c",
    show_cell=False,
    cellstyle="-w",
    **kwargs,
):
    ax = ax or plt.gca()
    if isinstance(nper, int):
        nperx, npery = nper, nper
    elif hasattr(nper, "__len__") and len(nper) == 2:
        nperx, npery = nper
    else:
        raise ValueError(f"Wrong type for nper: {nper}")

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
    lmax = max(lx, ly)
    delta = 0.1 * lmax
    ax.set_xlim(-delta, nperx * bv[0][0] + npery * bv[1][0] + delta)
    ax.set_ylim(-delta, nperx * bv[0][1] + npery * bv[1][1] + delta)
    if show_cell:
        plot_unit_cell(ax, bv, cellstyle)
    ax.set_aspect("equal")
    return ims


def plot_structure(
    sim, plotter=None, nper=(1, 1), dz=0.0, null_thickness=None, **kwargs
):
    if "layer_colors" in kwargs:
        layer_colors = kwargs.pop("layer_colors")
    else:
        layer_colors = None
    if "layer_metallic" in kwargs:
        layer_metallic = kwargs.pop("layer_metallic")
    else:
        layer_metallic = None
    if "layer_roughness" in kwargs:
        layer_roughness = kwargs.pop("layer_roughness")
    else:
        layer_roughness = None

    p = plotter or pyvista.Plotter()
    name = r"permittivity (Re)"
    null_thickness = null_thickness or bk.max([layer.thickness for layer in sim.layers])
    bvs = sim.lattice.basis_vectors

    transform_matrix = bk.array(
        [
            [bvs[0][0], bvs[1][0], 0, 0],
            [bvs[0][1], bvs[1][1], 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    for jx in range(nper[0]):

        for jy in range(nper[1]):
            z = 0
            x0, y0 = jx, jy

            for ilayer, layer in enumerate(sim.layers):
                thickness = layer.thickness
                if thickness == 0:
                    thickness = null_thickness
                    # if float(layer.epsilon.real) != 1:
                    #     thickness = null_thickness
                if layer.is_uniform:
                    if float(layer.epsilon.real) != 1:
                        grid = pyvista.UniformGrid()
                        grid.dimensions = (2, 2, 2)
                        grid.origin = (
                            x0,
                            y0,
                            z,
                        )  # The bottom left corner of the data set
                        grid.spacing = (
                            1,
                            1,
                            thickness,
                        )  # These are the cell sizes along each axis
                        val = layer.epsilon.real
                        grid.cell_data[name] = bk.array(val)
                        mesh = grid.extract_surface()
                        mesh = mesh.transform(transform_matrix)
                        if layer_colors is not None:
                            kwargs["color"] = layer_colors[ilayer]
                        if layer_metallic is not None:
                            kwargs["metallic"] = layer_metallic[ilayer]

                        if layer_roughness is not None:
                            kwargs["roughness"] = layer_roughness[ilayer]

                        p.add_mesh(mesh, **kwargs)
                        # p.add_mesh(
                        #     mesh,
                        #     metallic=0.3,
                        #     roughness=0.1,
                        #     pbr=True,
                        #     diffuse=1,
                        #     color=1 - 0.1 * bk.random.rand(3),
                        # )
                else:
                    try:
                        epsgrid = layer.epsilon.real
                    except Exception:
                        epsgrid = layer.epsilon
                    Nx, Ny = epsgrid.shape
                    # values = bk.reshape(epsgrid, (Nx, Ny, 1))
                    epsgrid = epsgrid.T
                    values = epsgrid[:, :, None]
                    # Create the spatial reference
                    grid = pyvista.UniformGrid()
                    grid.dimensions = bk.array(values.shape) + 1
                    grid.origin = (x0, y0, z)  # The bottom left corner of the data set
                    grid.spacing = (1 / Nx, 1 / Ny, thickness)
                    grid.cell_data[name] = values.flatten()  # Flatten the array!
                    vals = bk.unique(epsgrid)
                    for ival, v in enumerate(vals):
                        if v != 1:
                            threshed = grid.threshold([v - 1e-7 * v, v + 1e-7 * v])

                            threshed = threshed.transform(transform_matrix)

                            if layer_colors is not None:
                                kwargs["color"] = layer_colors[ilayer][ival]

                            if layer_metallic is not None:
                                kwargs["metallic"] = layer_metallic[ilayer][ival]
                            if layer_roughness is not None:
                                kwargs["roughness"] = layer_roughness[ilayer][ival]
                            p.add_mesh(threshed, **kwargs)

                            # p.add_mesh(
                            #     threshed,
                            #     metallic=0.3,
                            #     roughness=0.1,
                            #     pbr=True,
                            #     diffuse=1,
                            #     color=colors[1],
                            #     opacity=opacity,
                            # )

                z += thickness + dz
    p.show_axes()
    p.camera.azimuth -= 180
    p.camera.elevation -= 180
    p.camera.roll += 180
    return p
