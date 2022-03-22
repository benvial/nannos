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
#
# np.random.seed(1984)


#############################################################################
# Set a backend supporting automatic differentiation

# nn.set_backend("autograd")
nn.set_backend("torch")
# nn.use_gpu(True)
no = nn.optimize

bk = nn.backend
formulation = "original"
nh = 51
L1 = [1.5, 0]
L2 = [0, 1.5]
rat_unit_cel = L1[0] / L2[1]
theta = 0.0 * bk.pi / 180
phi = 0.0 * bk.pi / 180
psi = 0.0 * bk.pi / 180

Nx = 2**6
Ny = 2**6

eps_sup = 1
eps_sub = 1.0
eps_min = 1.0
eps_layer = (3) ** 2 + 0.01j
eps_max = 2.60**2 + 0.01j

h_layer = 0.43

h_ms = 0.7

rfilt = Nx / 30
maxiter = 20
order_target = (0, 0)
freq_target = 1 / 3

Nlayers = 9

rfilt0 = 10 * rfilt  # Nx / 12
stopval = -0.49


def run(density, proj_level=None, rfilt=0, freq=1, nh=nh, psi=0, nn=nn):

    lattice = nn.Lattice((L1, L2))
    pw = nn.PlaneWave(frequency=freq, angles=(theta, phi, psi))
    sup = nn.Layer("Superstrate", epsilon=eps_sup)  # input medium
    lay = nn.Layer(
        "Layer", epsilon=eps_layer, thickness=h_layer
    )  # actual the substrate
    sub = nn.Layer("Substrate", epsilon=eps_sub)  # output medium
    stack = [sup]

    density = bk.reshape(density, (Nlayers, Nx, Ny))
    for i, dens in enumerate(density):
        density_f = no.apply_filter(dens, rfilt)
        density_fp = (
            no.project(density_f, proj_level) if proj_level is not None else density_f
        )
        epsgrid = no.simp(density_fp, eps_min, eps_max, p=1)
        ms = nn.Layer(f"Metasurface_{i}", epsilon=1, thickness=h_ms / Nlayers)
        pattern = nn.Pattern(epsgrid, name="design")
        ms.add_pattern(pattern)
        stack.append(ms)
    stack += [lay, sub]
    sim = nn.Simulation(lattice, stack, pw, nh, formulation=formulation)
    return sim


##############################################################################
# Define objective function


def fun(density, proj_level, rfilt):
    sim_TE = run(density, proj_level, rfilt, freq=freq_target, psi=0)
    _, T_TE = sim_TE.diffraction_efficiencies(orders=True)

    sim_TM = run(density, proj_level, rfilt, freq=freq_target, psi=nn.pi / 2)
    _, T_TM = sim_TM.diffraction_efficiencies(orders=True)
    tar_TE = sim_TE.get_order(T_TE, order_target)
    tar_TM = sim_TM.get_order(T_TM, order_target)
    try:
        print(float(tar_TE), float(tar_TM))
    except:
        pass
    # return (tar_TE-tar_TM)/(tar_TE+tar_TM)
    d = (tar_TE) / (tar_TM)
    alpha = 0.5
    return tar_TE * alpha - (1 - alpha) * tar_TM
    # d = (tar_TE) - (tar_TM)
    # return bk.log(d)
    # return d
    # return bk.log(tar_TE* (1-tar_TM))
    # return bk.log(abs((1-tar_TM))) + tar_TE


##############################################################################
# Define initial density


def imshow(s, *args, **kwargs):
    extent = (0, rat_unit_cel, 0, 1)
    if nn.DEVICE == "cuda":
        plt.imshow(s.T.cpu(), *args, extent=extent, **kwargs)
    else:
        plt.imshow(s.T, *args, extent=extent, **kwargs)


densities0 = []
densities_plot0 = []

density0rand = np.random.rand(Nlayers, Nx, Ny)
for i in range(Nlayers):
    density0 = bk.array(density0rand)[i]
    density0 = 0.5 * (density0 + bk.fliplr(density0))
    density0 = 0.5 * (density0 + bk.flipud(density0))
    density0 = 0.5 * (density0 + density0.T)
    density0 = no.apply_filter(density0, rfilt0)
    density0 = (density0 - density0.min()) / (density0.max() - density0.min())
    #
    ### circle
    x, y = bk.linspace(0, 1, Nx), bk.linspace(0, 1, Ny)
    x, y = bk.meshgrid(x, y)
    density0 = bk.ones((Nx, Ny))

    R0 = 0.05
    density0[(x - 0.5) ** 2 + (y - 0.5) ** 2 < R0**2] = 0
    density0 = no.apply_filter(density0, rfilt0)
    density0 = (density0 - density0.min()) / (density0.max() - density0.min())

    # density0 *= 0.5

    # density0 = 1-density0

    # #### uniform
    ### need to add random because if uniform gradient is NaN with torch  (MKL error)
    # density0 = bk.ones((Nx, Ny))*0.5 + 0.0001*np.random.rand(Nx, Ny)

    density0 = density0.flatten()
    density_plot0 = bk.reshape(density0, (Nx, Ny))

    densities0.append(density0)
    densities_plot0.append(density_plot0)

density0 = bk.hstack(densities0)

fig, ax = plt.subplots(3, 3, figsize=(3, 3))

ax = ax.flatten()
for i, dens in enumerate(densities_plot0):
    a = ax[i]
    a.clear()
    plt.sca(a)
    imshow(dens, cmap="Blues")
    a.axis("off")
plt.suptitle(f"initial density")


# plt.close("all")
##############################################################################
# Define calback function

it = 0
fig, ax = plt.subplots(3, 3, figsize=(3, 3))
ax = ax.flatten()


def callback(x, y, proj_level, rfilt):
    global it
    print(f"iteration {it}")
    density = bk.reshape(x, (Nlayers, Nx, Ny))

    for i, dens in enumerate(density):
        a = ax[i]
        density_f = no.apply_filter(dens, rfilt)
        density_fp = no.project(density_f, proj_level)
        # plt.figure()
        a.clear()
        plt.sca(a)
        imshow(density_fp, cmap="Blues")
        a.axis("off")
        # plt.colorbar()
    plt.suptitle(f"iteration {it}, objective = {y:.5f}")
    # plt.tight_layout()
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
    stopval=stopval,
    args=(1, rfilt),
    callback=callback,
    options={},
)

##############################################################################
# Run the optimization

density_opt, f_opt = opt.minimize()

# density_opt , f_opt= density0,0
##############################################################################
# Postprocess to get a binary design


density_bins = []
density_opt = bk.reshape(bk.array(density_opt), (Nlayers, Nx, Ny))
proj_level = 2 ** (opt.threshold[-1] - 1)
for i in range(Nlayers):
    density_optf = no.apply_filter(density_opt[i], rfilt)
    density_optfp = no.project(density_optf, proj_level)
    density_bin = bk.ones_like(density_optfp)
    density_bin[density_optfp < 0.5] = 0
    density_bins.append(density_bin.flatten())


density_bin = bk.hstack(density_bins)

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


fig, ax = plt.subplots(3, 3, figsize=(3, 3))
ax = ax.flatten()

density_bin_plot = density_bin.reshape((Nlayers, Nx, Ny))

for i, dens in enumerate(density_bin_plot):
    a = ax[i]
    a.clear()
    plt.sca(a)
    imshow(dens, cmap="Blues")
    a.axis("off")
    # plt.colorbar()
plt.suptitle(f"iteration {it}, objective = {f_opt:.5f}")


import pyvista as pv

pv.set_plot_theme("document")
colors = ["#ed7559", "#4589b5", "#cad45f", "#7a6773", "#ed59da"]


def plot_struc(sim, p, nper=(1, 1), dz=0.0, opacity=1):

    bvs = sim.lattice.basis_vectors

    for jx in range(nper[0]):

        for jy in range(nper[1]):
            z = 0
            x0, y0 = bvs[0][0] * jx, bvs[1][1] * jy

            for layer in np.flipud(sim.layers):
                thickness = layer.thickness
                if thickness == 0:
                    if float(layer.epsilon).real != 1:
                        print("ssss")
                        thickness = 3
                if layer.is_uniform:
                    if thickness != 0:
                        grid = pv.UniformGrid()
                        grid.dimensions = (2, 2, 2)
                        grid.origin = (
                            x0,
                            y0,
                            0,
                            z,
                        )  # The bottom left corner of the data set
                        grid.spacing = (
                            bvs[0][0],
                            bvs[1][1],
                            thickness,
                        )  # These are the cell sizes along each axis
                        grid.cell_data["values"] = np.array(
                            [layer.epsilon.real]
                        )  # Flatten the array!
                        # grid.plot()
                        mesh = grid.extract_surface()
                        p.add_mesh(
                            mesh,
                            metallic=0.3,
                            roughness=0.1,
                            pbr=True,
                            diffuse=1,
                            color=1 - 0.1 * np.random.rand(3),
                        )
                else:
                    epsgrid = layer.patterns[0].epsilon.real
                    # values = np.linspace(0, 10, 1000).reshape((20, 5, 10))
                    values = np.reshape(epsgrid, (Nx, Ny, 1))
                    # Create the spatial reference
                    grid = pv.UniformGrid()
                    grid.dimensions = np.array(values.shape) + 1
                    grid.origin = (x0, y0, z)  # The bottom left corner of the data set
                    grid.spacing = (bvs[0][0] / Nx, bvs[1][1] / Ny, thickness)
                    grid.cell_data["values"] = values.flatten()  # Flatten the array!
                    vals = np.unique(epsgrid)
                    for v in vals:
                        if v != 1:
                            threshed = grid.threshold([v - 1e-7 * v, v + 1e-7 * v])
                            p.add_mesh(
                                threshed,
                                metallic=0.3,
                                roughness=0.1,
                                pbr=True,
                                diffuse=1,
                                color=colors[1],
                                opacity=opacity,
                            )

                z += thickness + dz

    return p


#
# p = pv.Plotter()
# p=plot_struc(sim,p,nper=(5,5),dz = 1,opacity=0.5)
# p.show()


wls = np.linspace(2.5, 3.5, 200)


@nn.parloop(n_jobs=20)
def spec(wl):
    import nannos as nn

    nn.set_backend("torch")
    bk = nn.backend
    TT = []
    for psi in [0, nn.pi / 2]:
        sim = run(density_bin, None, 0, freq=1 / wl, psi=psi, nn=nn)
        R, T = sim.diffraction_efficiencies(orders=True)
        print("Σ R = ", float(sum(R)))
        print("Σ T = ", float(sum(T)))
        print("Σ R + T = ", float(sum(R + T)))
        Ttarget = sim.get_order(T, order_target)
        print("")
        print(f"Target transmission in order {order_target}")
        print(f"===================================")
        print(f"T_{order_target} = ", float(Ttarget))

        TT.append(Ttarget)
    TT = bk.array(TT)
    return TT[::2], TT[1::2]


out = spec(wls)


TTE = bk.array([_[0] for _ in out])
TTM = bk.array([_[1] for _ in out])

plt.figure()
plt.plot(wls, TTE, label="TE")
plt.plot(wls, TTM, label="TM")
plt.legend()
plt.xlabel("Wavelength (mm)")
plt.ylabel("Transmission")


# layers_solved = []
# for layer in self.layers:
#     layer = self._build_matrix(layer)
#     # layer.solve_eigenproblem(layer.matrix)
#     if layer.is_uniform:
#         # layer.eigenvectors = bk.eye(self.nh*2)
#         layer.solve_uniform(self.omega, self.kx, self.ky, self.nh)
#     else:
#         layer.solve_eigenproblem(layer.matrix)
#
#     layers_solved.append(layer)
# self.layers = layers_solved
# self.is_solved = True


#
#
#
# sim = run(density_bin, None, 0, freq=freq_target, psi=psi)
#
# self = sim
#
# layers_structured = []
# for layer in self.layers:
#     if not layer.is_uniform:
#         layers_structured.append(layer)
#
#
# import time
#
# layer = layers_structured[1]
#
# layer = self._build_matrix(layer)
# matrix = layer.matrix
#
# def solve_structured_layers(layer,self=self):
#     t1 = nn.tic()
#     # time.sleep(1)
#     layer = self._build_matrix(layer)
#     layer.solve_eigenproblem(layer.matrix)
#     # layer.solve_eigenproblem(matrix)
#     nn.toc(t1)
#     return layer


# t1 = nn.tic()
# for layer in layers_structured:
#     solve_structured_layers(layer)
#
# nn.toc(t1)

# @nn.parloop(n_jobs=9)
# def fpara(layer):
#     layer = self._build_matrix(layer)
#     layer.solve_eigenproblem(layer.matrix)
#     return layer
#
#
# fpara = nn.parloop(n_jobs=9)(solve_structured_layers)
# t1 = nn.tic()
# layers_structured = fpara(layers_structured)
# nn.toc(t1)
#
#
#
#
#
# fpara = nn.parloop(n_jobs=1)(solve_structured_layers)
# t1 = nn.tic()
# layers_structured = fpara([layers_structured[0]])
# nn.toc(t1)
#
# t1 = nn.tic()
# solve_structured_layers(layers_structured[0])
# nn.toc(t1)
#
#
# def solve_para(self,n_jobs=1):
#
#     layers_structured = [layer for layer in self.layers if not layer.is_uniform]
#
#     @nn.parloop(n_jobs=n_jobs)
#     def solve_structured_layers(layer,self=self):
#         layer = self._build_matrix(layer)
#         layer.solve_eigenproblem(layer.matrix)
#         return layer
#     layers_structured = solve_structured_layers(layers_structured)
#
#
#     layers_solved = []
#     i=0
#     for layer in self.layers:
#         if layer.is_uniform:
#             layer = self._build_matrix(layer)
#             layer.solve_uniform(self.omega, self.kx, self.ky, self.nh)
#         else:
#             layer = layers_structured[i]
#             i+=1
#         layers_solved.append(layer)
#     self.layers = layers_solved
#     self.is_solved = True
