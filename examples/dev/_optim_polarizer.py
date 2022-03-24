#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


"""
Topology optimization
=====================

Design of a polarizer
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import nannos as nn


def savefig(dir, base, it):
    name = dir + "/" + base + str(it).zfill(4) + ".png"
    plt.savefig(name)


def pltpause(interval):
    backend = plt.rcParams["backend"]
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return


def symmetrize_pattern(dens, x=True, y=True, s8=False):
    dens = bk.array(dens)
    if x:
        dens = 0.5 * (dens + bk.fliplr(dens))
    if y:
        dens = 0.5 * (dens + bk.flipud(dens))
    if s8:
        dens = 0.5 * (dens + dens.T)
    return dens


def format_tensor(T):
    p = [t.item() for t in T]
    ps = [f"{_:.3f}" for _ in p]
    return "  ".join(ps)


plt.close("all")
plt.ion()

#############################################################################
# Set a backend supporting automatic differentiation
# backend="autograd"


backend = "torch"
nn.set_backend(backend)
# nn.use_gpu(True)
no = nn.optimize
bk = nn.backend

# ------ RCWA ------
formulation = "original"
nh = 7
# ------ lattice ------
L1 = [1.1, 0]
L2 = [0, 1.1]
rat_unit_cel = L1[0] / L2[1]
# ------ angles ------
theta = 0.0 * bk.pi / 180
phi = 0.0 * bk.pi / 180
psi = 0.0 * bk.pi / 180
# ------ patterns ------
Nx = 2**6
Ny = 2**6
Npad = 0  # int(Nx / 10)
Nlayers = 16
# ------ permittivity ------
eps_sup = 1.0
eps_sub = 1.0
eps_min = 1.0
# eps_max = 2.7 + 0.0001j
# eps_min = 2.7 + 0.0001j
eps_max = 6.5 + 0.0001j
# ------ thickness ------
hlays = 0.4
h_ms = hlays * Nlayers
# ------ optimization ------
rfilt = Nx / 50
maxiter = 20
stopval = None  # -0.48
threshold = (0, 8)
# ------ objective ------
order_target = (0, 0)
alpha_TE = 0.75
logopt = False
Nwl = 6
wls_opt = bk.linspace(2.5, 3.5, Nwl)
# wls_opt = bk.linspace(3, 3.25, Nwl)
# ------ initial values ------
# init = "circle"
init = "random"
init = "lines"
rfilt0 = 5 * rfilt  # Nx / 12
R0 = 0.25
invert0 = True
normalize0 = True
filter0 = True
symz0 = True
# ------ plot ------
nax = (
    int(Nlayers**0.5)
    if int(Nlayers**0.5) ** 2 == Nlayers
    else int(Nlayers**0.5) + 1
)
nax_x, nax_y = nax, nax
figsize = (3, 3)


hmin = 0.3
hmax = 0.3
amin = 1.0
amax = 1.0


def xtoparams(x):
    a = x[0]
    hs = x[1 : Nlayers + 1]
    dens = x[Nlayers + 1 :]
    hs = (hmax - hmin) * hs + hmin
    a = (amax - amin) * a + amin
    return a, hs, dens


def pad(dens, Npad):
    if Npad != 0:
        dens[:Npad, :] = 1
        dens[-Npad:, :] = 1
        dens[:, :Npad] = 1
        dens[:, -Npad:] = 1
    return dens


def run(density, proj_level=None, rfilt=0, freq=1, nh=nh, psi=0, nn=nn):

    a, hs, density = xtoparams(density)

    L1 = (a, 0)
    L2 = (0, a)
    lattice = nn.Lattice((L1, L2))
    pw = nn.PlaneWave(frequency=freq, angles=(theta, phi, psi))
    sup = nn.Layer("Superstrate", epsilon=eps_sup)  # input medium
    # lay = nn.Layer(
    #     "Layer", epsilon=eps_layer, thickness=h_layer
    # )  # actual the substrate
    sub = nn.Layer("Substrate", epsilon=eps_sub)  # output medium
    stack = [sup]

    density = bk.reshape(density, (Nlayers, Nx, Ny))
    for i, dens in enumerate(density):
        dens = symmetrize_pattern(dens)
        density_f = no.apply_filter(dens, rfilt)
        density_f = pad(density_f, Npad)
        density_fp = (
            no.project(density_f, proj_level) if proj_level is not None else density_f
        )
        epsgrid = no.simp(density_fp, eps_min, eps_max, p=1)
        ms = nn.Layer(f"Metasurface_{i}", epsilon=1, thickness=hs[i])
        pattern = nn.Pattern(epsgrid, name="design")
        ms.add_pattern(pattern)
        stack.append(ms)
    # stack += [lay, sub]
    stack += [sub]
    sim = nn.Simulation(lattice, stack, pw, nh, formulation=formulation)

    return sim


##############################################################################
# Define objective function


fig_obj, ax_obj = plt.subplots(1, 1, figsize=figsize)


def fun(density, proj_level, rfilt):
    global tars_TE
    global tars_TM

    n_jobs = 1

    tars_TE = []
    tars_TM = []

    a, hs, _ = xtoparams(density)

    @nn.parloop(n_jobs=n_jobs)
    def spec_TE(wl):
        import nannos as nn

        nn.set_backend(backend)
        bk = nn.backend
        sim_TE = run(density, proj_level, rfilt, freq=1 / wl, psi=0, nn=nn)
        _, T_TE = sim_TE.diffraction_efficiencies(orders=True)
        tar_TE = sim_TE.get_order(T_TE, order_target)
        return tar_TE

    @nn.parloop(n_jobs=n_jobs)
    def spec_TM(wl):
        import nannos as nn

        nn.set_backend(backend)
        bk = nn.backend
        sim_TM = run(density, proj_level, rfilt, freq=1 / wl, psi=nn.pi / 2)
        _, T_TM = sim_TM.diffraction_efficiencies(orders=True)
        tar_TM = sim_TM.get_order(T_TM, order_target)
        return tar_TM

    tars_TE = spec_TE(wls_opt)
    tars_TM = spec_TM(wls_opt)

    #
    # for wl in wls:
    #     # print(f"wavelength = ", wl)
    #     freq_target = 1 / wl
    #     sim_TE = run(density, proj_level, rfilt, freq=freq_target, psi=0)
    #     _, T_TE = sim_TE.diffraction_efficiencies(orders=True)
    #
    #     sim_TM = run(density, proj_level, rfilt, freq=freq_target, psi=nn.pi / 2)
    #     _, T_TM = sim_TM.diffraction_efficiencies(orders=True)
    #     tar_TE = sim_TE.get_order(T_TE, order_target)
    #     tar_TM = sim_TM.get_order(T_TM, order_target)
    #     tars_TE.append(tar_TE)
    #     tars_TM.append(tar_TM)

    if tars_TE[0].grad_fn is not None:
        print("wavelength = ", format_tensor(wls_opt))
        print("T TE       = ", format_tensor(tars_TE))
        print("T TM       = ", format_tensor(tars_TM))
    tar_TE = sum(tars_TE) / Nwl
    tar_TM = sum(tars_TM) / Nwl

    if tars_TE[0].grad_fn is not None:
        print(f"mean T TE = {float(tar_TE):0.3f}")
        print(f"mean T TM = {float(tar_TM):0.3f}")

        print("thicknesses = ", format_tensor(hs))
        print(f"period      = {a.item():0.3f}")

        print("------------------------------------------------")

    if logopt:
        out = alpha_TE * bk.log10(tar_TE) - (1 - alpha_TE) * bk.log10(tar_TM)
    else:
        out = alpha_TE * tar_TE - (1 - alpha_TE) * tar_TM

    return out


##############################################################################
# Define initial density


def imshow(s, *args, **kwargs):
    extent = (0, rat_unit_cel, 0, 1)
    if nn.DEVICE == "cuda":
        plt.imshow(s.T.cpu(), *args, vmin=0, vmax=1, extent=extent, **kwargs)
    else:
        plt.imshow(s.T, *args, vmin=0, vmax=1, extent=extent, **kwargs)


densities0 = []
densities_plot0 = []

for i in range(Nlayers):

    x, y = bk.linspace(0, 1, Nx), bk.linspace(0, 1, Ny)
    x, y = bk.meshgrid(x, y)
    density0 = bk.ones((Nx, Ny))
    if init == "random":

        density0rand = np.random.rand(Nx, Ny)
        # random
        # density0 = bk.array(density0rand)
        # density0 = 0.5 * (density0 + bk.fliplr(density0))
        # density0 = 0.5 * (density0 + bk.flipud(density0))
        # density0 = 0.5 * (density0 + density0.T)
        density0 = symmetrize_pattern(density0rand)
    elif init == "circle":
        ### circle
        density0[(x - 0.5) ** 2 + (y - 0.5) ** 2 < R0**2] = 0
        # density0 *= 0.5

    elif init == "lines":
        if i % 2:
            density0[abs(x - 0.5) < R0] = 0
        else:
            density0[abs(y - 0.5) < R0] = 0

        density0 += 0.0001 * np.random.rand(Nx, Ny)

    else:

        #### uniform
        ## need to add random because if uniform gradient is NaN with torch  (MKL error)
        density0 = bk.ones((Nx, Ny)) * 0.5 + 0.0001 * np.random.rand(Nx, Ny)

    if filter0:
        density0 = no.apply_filter(density0, rfilt0)

    if normalize0:
        density0 = (density0 - density0.min()) / (density0.max() - density0.min())

    if invert0:
        density0 = 1 - density0

    density0 = density0.flatten()
    density_plot0 = bk.reshape(density0, (Nx, Ny))

    densities0.append(density0)
    densities_plot0.append(density_plot0)

if symz0:
    densities0 = bk.reshape(bk.hstack(densities0), (Nlayers, Nx, Ny))
    densities0[int(Nlayers / 2) :] = bk.flipud(densities0[: -int(Nlayers / 2)])
    densities0 = list(densities0.ravel())


geompar0 = bk.array(np.random.rand(Nlayers + 1))

densities0 = [geompar0] + densities0

density0 = bk.hstack(densities0)
#
# fig, ax = plt.subplots(nax_x,nax_y, figsize=figsize)
#
# ax = ax.flatten()
# for i, dens in enumerate(densities_plot0):
#     a = ax[i]
#     a.clear()
#     plt.sca(a)
#     imshow(dens, cmap="Blues")
#     a.axis("off")
# plt.suptitle(f"initial density")


# plt.close("all")
##############################################################################
# Define calback function

it = 0
fig, ax = plt.subplots(nax_x, nax_y, figsize=figsize)
ax = ax.flatten()


def callback(x, y, proj_level, rfilt):
    global it
    print(f"iteration {it}")
    a, hs, density = xtoparams(x)

    density = bk.reshape(density, (Nlayers, Nx, Ny))

    for i, dens in enumerate(density):
        a = ax[i]
        dens = symmetrize_pattern(dens)
        density_f = no.apply_filter(dens, rfilt)

        density_f = pad(density_f, Npad)
        density_fp = no.project(density_f, proj_level)
        # plt.figure()
        a.clear()
        plt.sca(a)
        imshow(density_fp, cmap="Blues")
        a.axis("off")
        # plt.colorbar()
    for a in ax:
        a.axis("off")
    plt.suptitle(f"iteration {it}, objective = {y:.5f}")
    # plt.tight_layout()
    pltpause(0.1)

    savefig("/home/ben/Code/dev/prod/nannos/polarizer", "dens", it)

    # ---------

    nlines = 5
    if it > 0:
        for line in ax_obj.get_lines()[-2 * nlines :]:
            line.set_alpha(0.04)
        for line in ax_obj.get_lines()[: -2 * nlines]:
            line.remove()

    # ax_obj.clear()
    plt.sca(ax_obj)
    (line_TE,) = ax_obj.plot(wls_opt, bk.array(tars_TE), "o-r")
    (line_TM,) = ax_obj.plot(wls_opt, bk.array(tars_TM), "s-b")
    ax_obj.legend([line_TE, line_TM], ["TE", "TM"], loc="lower left")
    ax_obj.set_ylim(-0.1, 1.1)
    ax_obj.set_xlabel("Wavelength (mm)")
    ax_obj.set_ylabel("Transmission")
    pltpause(0.1)
    savefig("/home/ben/Code/dev/prod/nannos/polarizer", "obj", it)

    it += 1


##############################################################################
# Create `TopologyOptimizer` object


# x_opt = np.load("xopt.npz",allow_pickle=True)["x_opt"]
# density0 = x_opt

# threshold = (0,1)
# maxiter = 40

opt = no.TopologyOptimizer(
    fun,
    density0,
    method="nlopt",
    threshold=threshold,
    maxiter=maxiter,
    stopval=stopval,
    args=(1, rfilt),
    callback=callback,
    options={},
)

##############################################################################
# Run the optimization

x_opt, f_opt = opt.minimize()
# density_opt , f_opt= density0,0

# np.savez("xopt.npz",x_opt=x_opt)
# xsx
a, hs, density_opt = xtoparams(x_opt)

##############################################################################
# Postprocess to get a binary design


density_bins = []
density_opt = bk.reshape(bk.array(density_opt), (Nlayers, Nx, Ny))
proj_level = 2 ** (opt.threshold[-1] - 1)
for i in range(Nlayers):
    dens = symmetrize_pattern(density_opt[i])
    density_optf = no.apply_filter(dens, rfilt)
    density_optf = pad(density_optf, Npad)
    density_optfp = no.project(density_optf, proj_level)
    density_bin = bk.ones_like(density_optfp)
    density_bin[density_optfp < 0.5] = 0
    density_bins.append(density_bin.flatten())


density_bin = bk.hstack(density_bins)
#
# for psi in [0, nn.pi / 2]:
#     sim = run(density_bin, None, 0, freq=freq_target, psi=psi)
#     R, T = sim.diffraction_efficiencies(orders=True)
#     print("Σ R = ", float(sum(R)))
#     print("Σ T = ", float(sum(T)))
#     print("Σ R + T = ", float(sum(R + T)))
#     Ttarget = sim.get_order(T, order_target)
#     print("")
#     print(f"Target transmission in order {order_target}")
#     print(f"===================================")
#     print(f"T_{order_target} = ", float(Ttarget))
#

fig, ax = plt.subplots(nax_x, nax_y, figsize=figsize)
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

x_opt_bin = bk.array(x_opt.copy())
x_opt_bin[Nlayers + 1 :] = density_bin

sim = run(x_opt_bin, None, 0)


##############################################################################
# 3D plot


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
                    try:
                        epsgrid = layer.patterns[0].epsilon.real
                    except:
                        epsgrid = layer.patterns[0].epsilon
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


# #
# p = pv.Plotter()
# p = plot_struc(sim, p, nper=(5, 5), dz=0, opacity=1)
# p.show()


##############################################################################
# Spectra

wls = np.linspace(2.0, 4, 500)


@nn.parloop(n_jobs=20)
def spec(wl):
    import nannos as nn

    nn.set_backend(backend)
    bk = nn.backend
    TT = []
    for psi in [0, nn.pi / 2]:
        sim = run(x_opt_bin, None, 0, freq=1 / wl, psi=psi, nh=51, nn=nn)
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
plt.fill_between([wls_opt[0], wls_opt[-1]], [1, 1], alpha=0.1, color="#3e3e45", lw=0)
plt.ylim(0, 1)
plt.legend()
plt.xlabel("Wavelength (mm)")
plt.ylabel("Transmission")

plt.tight_layout()


# in GHz/ dB
freqs = nn.c / (wls * 1e-3) / 1e9
freqs_opt = nn.c / (wls_opt * 1e-3) / 1e9

TTE_dB = 20 * bk.log10(TTE)
TTM_dB = 20 * bk.log10(TTM)

plt.figure()
plt.plot(freqs, TTE_dB, label="TE")
plt.plot(freqs, TTM_dB, label="TM")
plt.fill_between(
    [freqs_opt[0], freqs_opt[-1]], [-35, -35], [3, 3], alpha=0.1, color="#3e3e45", lw=0
)
plt.ylim(-35, 3)
plt.legend()
plt.xlabel("Frequency (GHz)")
plt.ylabel(r"$S_{21}$")

plt.tight_layout()
