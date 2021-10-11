#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


"""
Topology optimization
=====================

Design of an anti-reflection metasurface.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import nlopt
import numpy as np
from autograd import grad
from scipy.optimize import minimize

import nannos as nn
from nannos.helpers import filter, project

plt.close("all")
plt.ion()

##############################################################################
# We will study a benchmark of hole in a dielectric surface

nn.set_backend("autograd")

nh = 50
L1 = [1.0, 0]
L2 = [0, 1.0]
freq = 0.6
theta = 0.0 * np.pi / 180
phi = 0.0 * np.pi / 180
psi = 0.0 * np.pi / 180

Nx = 2 ** 7
Ny = 2 ** 7

eps_sup = 1.0
eps_slab = 16.0
eps_sub = 4.0
eps_min = 1.0
eps_max = 4.0

h_slab = 1.0
h_ms = 0.5

rfilt = Nx / 25


def run(density, proj_level=None, rfilt=0, freq=1, nh=nh):
    metasurface = density is not None
    if metasurface:
        density = np.reshape(density, (Nx, Ny))
        density_f = filter(density, rfilt)
        density_fp = (
            project(density_f, proj_level) if proj_level is not None else density_f
        )
        epsgrid = (eps_max - eps_min) * density_fp + eps_min

    ##############################################################################
    # Define the lattice

    lattice = nn.Lattice((L1, L2))

    ##############################################################################
    # Define the incident plane wave

    pw = nn.PlaneWave(frequency=freq, angles=(theta, phi, psi))

    ##############################################################################
    # Define the layers

    sup = nn.Layer("Superstrate", epsilon=eps_sup)
    slab = nn.Layer("Slab", epsilon=eps_slab, thickness=h_slab)
    sub = nn.Layer("Substrate", epsilon=eps_sub)

    if metasurface:
        ##############################################################################
        # Define the pattern and add it to the metasurface layer
        ms = nn.Layer("Metasurface", epsilon=1, thickness=h_ms)
        pattern = nn.Pattern(epsgrid, name="design")
        ms.add_pattern(pattern)
        stack = [sup, ms, slab, sub]
    else:
        stack = [sup, slab, sub]

    ##############################################################################
    # Define the simulation

    simu = nn.Simulation(lattice, stack, pw, nh, formulation="original")

    return simu


##############################################################################
# Unpatterned

freqs = np.linspace(0.70, 1.3, 250)
Rslab = []

for freq in freqs:

    simu = run(density=None, freq=freq, nh=2)
    R, T = simu.diffraction_efficiencies()
    print(R)
    Rslab.append(R)

plt.figure()
plt.plot(freqs, Rslab, "or")
plt.pause(0.1)

freq_target = 0.81
freq_target = 1.06


def fun(density, proj_level, rfilt):
    simu = run(density, proj_level, rfilt, freq=freq_target)
    R, T = simu.diffraction_efficiencies()
    return R


density0 = np.random.rand(Nx, Ny)

density0 = 0.5 * (density0 + np.fliplr(density0))
density0 = 0.5 * (density0 + np.flipud(density0))
density0 = 0.5 * (density0 + np.transpose(density0))


density0 = filter(density0, Nx / 12)
density0 = (density0 - density0.min()) / (density0.max() - density0.min())


density0 = density0.flatten()
grad_fun = grad(fun)


df_ddensity = grad_fun(density0, proj_level=1, rfilt=rfilt)


density_plot0 = np.reshape(density0, (Nx, Ny))
df_ddensity_plot0 = np.reshape(df_ddensity, (Nx, Ny))

plt.figure()
plt.imshow(density_plot0)
plt.figure()
plt.imshow(df_ddensity_plot0)


def callback(x, y, rfilt, proj_level):
    density = np.reshape(x, (Nx, Ny))
    density_f = filter(density, rfilt)
    density_fp = project(density_f, proj_level)

    plt.clf()
    plt.imshow(density_fp)
    plt.axis("off")
    plt.colorbar()
    plt.title(f"objective = {y}")
    plt.tight_layout()
    plt.show()
    plt.pause(0.1)


class StopFunError(Exception):
    """Raised when stop value reached"""

    pass


class StopFun:
    def __init__(self, fun, stopval=None):
        self.fun_in = fun
        self.stopval = stopval

    def fun(self, x, *args):
        self.f = self.fun_in(x, *args)
        if self.stopval is not None:
            if self.f < self.stopval:
                raise StopFunError("Stop value reached.")
        self.x = x
        return self.f


class TopologyOptimizer:
    def __init__(
        self,
        fun,
        x0,
        method="scipy",
        threshold=(0, 8),
        maxiter=10,
        stopval=None,
        args=None,
        callback=None,
        options={},
    ):
        self.fun = fun
        self.x0 = x0
        self.nvar = len(x0)
        self.method = method
        self.threshold = threshold
        self.maxiter = maxiter
        self.stopval = stopval
        self.args = args
        self.callback = callback
        self.options = options
        self.grad_fun = grad(fun)

    def min_function(self):
        f = self.fun(x)

    def minimize(self):

        print("#################################################")
        print(f"Topology optimization with {self.nvar} variables")
        print("#################################################")
        print("")
        for iopt in range(*self.threshold):
            print(f"global iteration {iopt}")
            print("-----------------------")

            proj_level = 2 ** iopt
            args = list(self.args)
            args[1] = proj_level
            args = tuple(args)
            if self.method == "scipy":

                def fun_scipy(x, *args):
                    y = fun(x, *args)
                    print(f"current value = {y}")
                    if self.callback is not None:
                        self.callback(x, y, *args)
                    return y

                bounds = [(0, 1) for _ in range(self.nvar)]
                options = {"maxiter": self.maxiter}
                if self.options is not None:
                    options.update(self.options)

                stop = StopFun(fun=fun_scipy, stopval=self.stopval)
                try:
                    opt = minimize(
                        stop.fun,
                        self.x0,
                        args=args,
                        method="L-BFGS-B",
                        jac=self.grad_fun,
                        bounds=bounds,
                        options=options,
                    )
                    xopt = opt.x
                    fopt = opt.fun
                except StopFunError as e:
                    print(e)
                    xopt = stop.x
                    fopt = stop.f

            else:

                def fun_nlopt(x, gradn):
                    gradn[:] = self.grad_fun(x, *args)
                    y = fun(x, *args)
                    print(f"current value = {y}")
                    if self.callback is not None:
                        self.callback(x, y, *args)
                    return y

                lb = np.zeros(self.nvar, dtype=float)
                ub = np.ones(self.nvar, dtype=float)

                opt = nlopt.opt(nlopt.LD_MMA, self.nvar)
                opt.set_lower_bounds(lb)
                opt.set_upper_bounds(ub)

                # opt.set_ftol_rel(1e-16)
                # opt.set_xtol_rel(1e-16)
                if self.stopval is not None:
                    opt.set_stopval(self.stopval)
                if self.maxiter is not None:
                    opt.set_maxeval(self.maxiter)

                opt.set_min_objective(fun_nlopt)
                xopt = opt.optimize(self.x0)
                fopt = opt.last_optimum_value()
            self.x0 = xopt
        return xopt, fopt


opt = TopologyOptimizer(
    fun,
    density0,
    method="nlopt",
    threshold=(0, 8),
    maxiter=20,
    stopval=0.0,
    args=(rfilt, 1),
    callback=callback,
    options={},
)

plt.close("all")
density_opt, Ropt = opt.minimize()

density_opt = np.reshape(density_opt, (Nx, Ny))
proj_level = 2 ** (opt.threshold[-1] - 1)
density_optf = filter(density_opt, rfilt)
density_optfp = project(density_optf, proj_level)

density_bin = np.ones_like(density_optfp)
density_bin[density_optfp < 0.5] = 0


simu = run(density_bin, None, 0, freq=freq_target)
R, T = simu.diffraction_efficiencies()
print(R, T)

plt.clf()
plt.imshow(density_bin)
plt.axis("off")
plt.colorbar()
plt.title(f"objective = {R}")
plt.tight_layout()
plt.show()
plt.pause(0.1)


freqs_ms = np.linspace(0.9, 1.1, 100)
Rms = []

for freq in freqs_ms:
    simu = run(density=density_bin, proj_level=None, rfilt=0, freq=freq, nh=nh)
    R, T = simu.diffraction_efficiencies()
    print(R)
    Rms.append(R)


plt.figure()
plt.plot(freqs, Rslab, "r--")
plt.plot(freqs_ms, Rms, "b")
plt.pause(0.1)


simu = run(density=density_bin, proj_level=None, rfilt=0, freq=freq_target, nh=nh)
R, T = simu.diffraction_efficiencies()
print(R)
