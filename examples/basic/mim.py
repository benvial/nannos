#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


"""
MIM
======================

Reflection spectrum.
"""


# sphinx_gallery_thumbnail_number = -1


import matplotlib.pyplot as plt
import numpy as np
import pyvista
import refidx as ri
from matplotlib.colors import ListedColormap

import nannos as nn

no = nn.optimize
bk = np


plt.ion()
plt.close("all")
#########################################################################


def symmetrize(dens, x=False, y=False):
    if y == True:
        dens = 0.5 * (dens + bk.fliplr(dens))
    if x == True:
        dens = 0.5 * (dens + bk.flipud(dens))
    return dens


db = ri.DataBase()
Ag = db.materials["main"]["Ag"]["Johnson"]
Au = db.materials["main"]["Au"]["Johnson"]


N = 2**7
period = 0.2
l_cube = 0.05
h_cube = l_cube
h_slab = 1e-3


#########################################################################
# Define the lattice

lattice = nn.Lattice(([period, 0], [0, period]), discretization=(N, N))


# np.random.seed(11)
Nx, Ny = lattice.discretization
rfilt = Nx / 10
density0 = np.random.rand(Nx, Ny)
density0 = bk.array(density0)
density0 = symmetrize(density0, x=True)

density0 = no.apply_filter(density0, rfilt)
# density0 = no.project(density0, 111111)
density0 = (density0 - density0.min()) / (density0.max() - density0.min())

density0[density0 > 0.5] = 1
density0[density0 <= 0.5] = 5


def simu(wl):

    eps_Ag = Ag.get_index(wl) ** 2
    eps_Au = Au.get_index(wl) ** 2
    eps_slab = 1.45**2

    #########################################################################
    # Define the layer with the nanostructure

    epsilon = lattice.ones()
    square = lattice.square(center=(0.5 * period, 0.5 * period), width=l_cube)
    epsilon[square] = eps_Ag.conj()
    struct_layer = lattice.Layer("Struct", thickness=h_cube)
    struct_layer.epsilon = epsilon
    # struct_layer.epsilon = density0

    #########################################################################
    # Define the simulation

    slab = lattice.Layer("Slab", thickness=h_slab, epsilon=eps_slab)
    sup = lattice.Layer("Superstrate", epsilon=1)
    # sub1 = lattice.Layer("Substrate1", epsilon=eps_Si.conj(), thickness=1112)
    sub = lattice.Layer("Substrate", epsilon=eps_Au.conj())
    stack = [sup, struct_layer, slab, sub]
    # stack = [sup, sub1, sub]
    # stack = [sup, sub]

    pw = nn.PlaneWave(wavelength=wl, angles=(0, 0, 0))
    sim = nn.Simulation(stack, pw, nh)  # ), formulation="tangent")
    return sim


nh = 50

wls = np.linspace(0.5, 1, 1)
Rs = []
for wl in wls:
    sim = simu(wl)

    R, T = sim.diffraction_efficiencies()

    print(wl, R)

    Rs.append(R)


# cmap = ListedColormap(["#dddddd", "#73a0e8"])

# plt.figure()
# im = struct_layer.plot(cmap=cmap)
# cbar = plt.colorbar(im[0], ticks=[1, eps_Au.real])
# plt.xlabel(r"$x$ ($\mu$m)")
# plt.ylabel(r"$y$ ($\mu$m)")
# plt.title(r"$\varepsilon$")
# plt.axis("scaled")
# plt.tight_layout()
# plt.show()


wls = np.array(wls)

plt.figure()
plt.plot(wls * 1000, Rs, c="#be4c83")
# plt.ylim(0.0, 1)
plt.xlabel(r"$\lambda$ (nm)")
plt.ylabel("$R$")
plt.tight_layout()

wl = 0.565
# wl = 0.767


# #########################################################################
# # Plot the fields at the resonant frequency

layer_prop = dict(color={}, metalic={}, roughness={}, opacity={})


def translatexy(mesh, x, y, bvs):
    t = x * bvs[0] + y * bvs[1]
    transform_matrix = bk.array(
        [
            [1, 0, 0, t[0]],
            [0, 1, 0, t[1]],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return mesh.transform(transform_matrix)


def _process_layer_prop(kwargs, layer_prop, layer, ival=None):
    for par in ["color", "metallic", "roughness", "opacity"]:
        if par not in layer_prop.keys():
            layer_prop[par] = {}

    if layer.name in layer_prop["color"].keys():
        color = layer_prop["color"][layer.name]
        if ival is not None:
            color = color[ival]
    else:
        color = "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])
    kwargs["color"] = color

    def _assign(kwargs, par, val=None):
        if layer.name in layer_prop[par].keys():
            tmp = layer_prop[par][layer.name]
            if ival is not None:
                tmp = tmp[ival]
        else:
            tmp = random.random() if val is None else val
        kwargs[par] = tmp
        return kwargs

    kwargs = _assign(kwargs, "metallic")
    kwargs = _assign(kwargs, "roughness")
    kwargs = _assign(kwargs, "opacity", 1)
    return kwargs


def plot_structure(
    sim,
    plotter=None,
    nper=(1, 1),
    dz=0.0,
    null_thickness=None,
    map=True,
    layer_prop=dict(color={}, metalic={}, roughness={}),
    **kwargs,
):
    if "cmap" not in kwargs.keys():
        kwargs["cmap"] = "inferno"

    bvs = sim.lattice.basis_vectors
    bvs = bk.array(bvs)

    p = plotter or pyvista.Plotter()
    name = r"permittivity (Re)"
    null_thickness = null_thickness or bk.max([layer.thickness for layer in sim.layers])

    transform_matrix = bk.array(
        [
            [bvs[0][0], bvs[1][0], 0, 0],
            [bvs[0][1], bvs[1][1], 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    x0, y0 = 0, 0
    z = 0
    for ilayer, layer in enumerate(sim.layers):
        thickness = layer.thickness
        if thickness == 0:
            thickness = null_thickness
        if layer.is_uniform:
            if float(layer.epsilon.real) != 1:
                grid = pyvista.ImageData()
                grid.dimensions = (2, 2, 2)
                grid.origin = (x0, y0, z)
                grid.spacing = (1, 1, thickness)
                val = layer.epsilon.real
                grid.cell_data[name] = bk.array(val)
                mesh = grid.extract_surface()
                mesh = mesh.transform(transform_matrix)
                mesh0 = pv.PolyData(mesh)
                if not map:
                    kwargs = _process_layer_prop(kwargs, layer_prop, layer)

                for jx in range(nper[0]):
                    for jy in range(nper[1]):
                        umesh = translatexy(mesh0.copy(), jx, jy, bvs)
                        p.add_mesh(umesh, **kwargs)

        else:
            try:
                epsgrid = layer.epsilon.real
            except Exception:
                epsgrid = layer.epsilon
            Nx, Ny = epsgrid.shape
            values = epsgrid[:, :, None]
            Nz = 1
            values = bk.repeat(epsgrid[:, :, bk.newaxis], Nz, axis=2)
            vals = bk.unique(epsgrid)
            for ival, v in enumerate(vals):
                if v != 1:

                    # Create the spatial reference
                    grid = pyvista.ImageData()
                    grid.dimensions = bk.array(values.shape) + 1
                    grid.origin = (x0, y0, z)  # The bottom left corner of the data set
                    grid.spacing = (1 / (Nx), 1 / (Ny), thickness / Nz)
                    grid.cell_data[name] = values.flatten(
                        order="F"
                    )  # Flatten the array!

                    threshed = grid.threshold([v - 1e-7 * abs(v), v + 1e-7 * abs(v)])
                    threshed = threshed.extract_surface()
                    # threshed = threshed.extract_geometry()
                    threshed0 = threshed.transform(transform_matrix)
                    # # threshed=threshed.triangulate()
                    # # threshed = threshed.decimate(0.75)
                    # # threshed = threshed.smooth(n_iter=20)
                    # # threshed= threshed.smooth_taubin(n_iter=50, pass_band=0.05)
                    # threshed0 = pv.PolyData(threshed)
                    # # threshed0 = threshed0.extract_geometry().clean(tolerance=1e-6)
                    if not map:
                        kwargs = _process_layer_prop(kwargs, layer_prop, layer, ival)

                    for jx in range(nper[0]):
                        for jy in range(nper[1]):
                            threshed = translatexy(threshed0.copy(), jx, jy, bvs)
                            p.add_mesh(threshed, **kwargs)

        z += thickness + dz

    light = pv.Light(position=(-z, -z, -z), focal_point=(0.0, 0.0, 0.0))
    p.add_light(light)
    p.show_axes()
    p.camera.azimuth -= 180
    p.camera.elevation -= 180
    p.camera.roll += 180
    return p


sim = simu(wl)


layer_prop = {}
layer_prop["color"] = dict(
    Substrate="#dfb733", Slab="#80ccbd", Struct=["#b6b19f", None]
)
layer_prop["metallic"] = dict(Substrate=1.0, Slab=0, Struct=[0, 0.9])
layer_prop["roughness"] = dict(Substrate=0.10, Slab=1, Struct=[0, 0.22])
layer_prop["opacity"] = dict(Slab=0.8)
# layer_prop["opacity"] = dict(Substrate=0.5,Slab=0.5, Struct=[0.5,0.5])

Nz = 20

zdata = bk.linspace(0, h_cube, Nz)
Edata, H = sim.get_field_grid("Struct", shape=(N, N), z=zdata)


Edata = H


nper = 7, 7

#########################################################################
# Electric field
bvs = sim.lattice.basis_vectors
bvs = bk.array(bvs)

from pyvista import examples

texture = examples.download_sky_box_cube_map()


null_thickness = bk.max([layer.thickness for layer in sim.layers])
layer = sim.layers[1]

transform_matrix = bk.array(
    [
        [bvs[0][0], bvs[1][0], 0, 0],
        [bvs[0][1], bvs[1][1], 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)

pv = pyvista
pl = pv.Plotter()
pl = plot_structure(
    sim,
    nper=nper,
    layer_prop=layer_prop,
    map=False,
    pbr=True,
    smooth_shading=False,
    plotter=pl,
)
# pl.show_axes()

texture = examples.download_cubemap_space_4k()
light = pv.Light(position=(-10, -10, -11), focal_point=(0.0, 0.0, 0.0))
pl.add_light(light)


import numpy as np

arr = np.array(
    [
        [200, 200, 200],
        [100, 100, 100],
        [100, 100, 100],
        [100, 100, 100],
    ],
    dtype=np.uint8,
)
arr = arr.reshape((2, 2, 3))
texture = pv.Texture(arr)


pl.set_environment_texture(texture)
pl.hide_axes()
# pl.show()


x0, y0, z0 = 0, 0, null_thickness + h_slab
Nx, Ny = sim.lattice.discretization
spl = int(Nx / 2**5)


Edata = Edata[:, ::spl, ::spl]
thickness = 0
Enorm = bk.linalg.norm(Edata, axis=0)
maxEnorm = bk.max(Enorm)

for i in range(Nz):
    E = Edata[:, :, :, i]

    grid = pyvista.ImageData()
    Nx, Ny = E.shape[1], E.shape[2]
    values = bk.ones((Nx, Ny, 1))
    grid.dimensions = bk.array(values.shape)
    grid.origin = (x0, y0, z0 + zdata[i])  # The bottom left corner of the data set
    grid.spacing = (1 / Nx, 1 / Ny, 1)

    grid = grid.extract_surface()

    Enorm = bk.linalg.norm(E, axis=0)

    scale = 0.01 / maxEnorm

    vectors = np.vstack(
        (
            E[0].real.ravel(),
            E[1].real.ravel(),
            E[2].real.ravel(),
        )
    ).T
    vectors /= maxEnorm

    scalars = Enorm.real.ravel()

    # add and scale
    grid["vectors"] = vectors * scale
    grid["scalars"] = scalars / maxEnorm
    mask = grid["scalars"] < 0.12
    grid["scalars"][mask] = 0  # null out smaller vectors
    grid["scalars"] *= maxEnorm
    # grid.set_active_vectors("vectors")

    grid = grid.transform(transform_matrix)

    # Make a geometric object to use as the glyph
    geom = pv.Arrow()  # This could be any dataset

    # Perform the glyph
    glyphs = grid.glyph(orient="vectors", scale="scalars", factor=scale, geom=geom)

    for jx in range(nper[0]):
        for jy in range(nper[1]):
            glyphs1 = translatexy(glyphs.copy(), jx, jy, bvs)

            pl.add_mesh(glyphs1, show_scalar_bar=False, lighting=False, cmap="coolwarm")
    # grid.arrows.plot()

cpos = pl.show(return_cpos=True)

xsx
# epsilon = sim.layers[1].epsilon

# extent = [0, period, 0, period]
# x, y = np.linspace(0, period, N), np.linspace(0, period, N)

# plt.figure()
# plt.imshow(epsilon.real, cmap="Greys", origin="lower", extent=extent)
# plt.imshow(nE2, alpha=0.9, origin="lower", extent=extent)
# plt.colorbar()
# s = 3
# plt.quiver(x[::s], y[::s], Ex[::s, ::s].real, Ey[::s, ::s].real, color="w")
# plt.xlabel(r"$x$ ($\mu$m)")
# plt.ylabel(r"$y$ ($\mu$m)")
# plt.title("$E$")
# plt.tight_layout()
# plt.show()

# #########################################################################
# # Magnetic field

# plt.figure()
# plt.imshow(epsilon.real, cmap="Greys", origin="lower", extent=extent)
# plt.imshow(nH2, alpha=0.9, origin="lower", extent=extent)
# plt.colorbar()
# plt.quiver(x[::s], y[::s], Hx[::s, ::s].real, Hy[::s, ::s].real, color="w")
# plt.xlabel(r"$x$ ($\mu$m)")
# plt.ylabel(r"$y$ ($\mu$m)")
# plt.title("$H$")
# plt.tight_layout()
# plt.show()
