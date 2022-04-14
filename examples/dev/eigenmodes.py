#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

import matplotlib.pyplot as plt
import numpy as np
import tetrachotomy as tc

import nannos as nn

# nn.set_backend("autograd")

plt.ion()

bk = nn.backend
formulation = "original"
nh = 1

lattice = nn.Lattice([[1.0, 0], [0, 1.0]], discretization=2**9)
sup = lattice.Layer("Superstrate", epsilon=1)
sub = lattice.Layer("Substrate", epsilon=1)
o = lattice.ones()
hole = lattice.circle(center=(0.5, 0.5), radius=0.2)

epsilon = 4 * o
# epsilon[hole] = 1
epsilon = 4
ms = lattice.Layer("Metasurface", thickness=3.5, epsilon=epsilon)

h = ms.thickness
eps = 4  # ms.epsilon

alpha = (eps**0.5 + 1) / (eps**0.5 - 1)
n = np.array(range(0, 10))
ev = (nn.pi * n - 1j * np.log(alpha)) / (2 * nn.pi * h * eps**0.5)

freqs = np.linspace(1.3, 1.5, 55)
#

freqs = np.linspace(-0.1, 0.1, 55)


freqs_re = np.linspace(0.1, 0.5, 60)
freqs_im = np.linspace(-0.05, 0.05, 50)
#
# plt.clf()
#
# DET = []
# for fre in freqs_re:
#     det_=[]
#     for fim in freqs_im:
#         f = fre + fim*1j
#         print(f)
#         # pw = nn.PlaneWave(wavelength=1/f - 0.1*1j, angles=(0, 0, 0))
#         pw = nn.PlaneWave(wavelength=1/f, angles=(0, 0, 0))
#         sim = nn.Simulation([sup, ms, sub], pw, nh=nh, formulation=formulation)
#         S = sim.get_S_matrix()[0][1]
#         # S = nn.helpers.block(S)
#         det = bk.linalg.det(S)
#
#         det_.append(det)
#
#         # plt.plot(f, bk.abs(det), "or")
#         # plt.pause(0.1)
#
#     DET.append(det_)
#
#
# DET = bk.array(DET).T
# plt.pcolor(freqs_re,freqs_im,abs(DET))
# plt.plot(ev.real,ev.imag,"or")
#
#
# xsx
import scipy

x0 = [0.2, 0]

x0 = [ev[2].real, ev[2].imag]


def func(x):
    f = x[0] + 1j * x[1]
    pw = nn.PlaneWave(wavelength=1 / f, angles=(0, 0, 0 * nn.pi / 2))
    sim = nn.Simulation([sup, ms, sub], pw, nh=nh, formulation=formulation)
    S = sim.get_S_matrix()[0][1]
    # S = nn.helpers.block(S)
    det = bk.linalg.det(S)
    out = 1 / (det)
    # print(f, out)
    return out.real, out.imag


options = {
    "col_deriv": 0,
    "xtol": 1.49012e-6,
    "ftol": 1.49012e-6,
    "gtol": 0.0,
    "maxiter": 200,
    "eps": 0.0,
    "factor": 100,
    "diag": None,
}
# res = scipy.optimize.root(
#     func, x0, args=(), method="hybr", jac=None, tol=None, callback=None, options=options
# )
#
# x0 = 1.4
# res = scipy.optimize.newton(
#     func,
#     x0,
#     fprime=None,
#     args=(),
#     tol=1.48e-08,
#     maxiter=250,
#     fprime2=None,
#     x1=None,
#     rtol=0.0,
#     full_output=False,
#     disp=True,
# )
#
# ‘Nelder-Mead’ (see here)
#
# ‘Powell’ (see here)
#
# ‘CG’ (see here)
#
# ‘BFGS’ (see here)
#
# ‘Newton-CG’ (see here)
#
# ‘L-BFGS-B’ (see here)
#
# ‘TNC’ (see here)
#
# ‘COBYLA’ (see here)
#
# ‘SLSQP’ (see here)
#
# ‘trust-constr’(see here)
#
# ‘dogleg’ (see here)
#
# ‘trust-ncg’ (see here)
#
# ‘trust-exact’ (see here)
#
# ‘trust-krylov’ (see here)

x0 = [1.2 * ev[3].real, 0]
x0 = [1 * ev[3].real, 0 * ev[3].imag]


def func(x):
    f = x[0] + 1j * x[1]
    pw = nn.PlaneWave(wavelength=1 / f, angles=(0, 0, 0 * nn.pi / 2))
    sim = nn.Simulation([sup, ms, sub], pw, nh=nh, formulation=formulation)
    S = sim.get_S_matrix()[0][1]
    # S = nn.helpers.block(S)
    det = bk.linalg.det(S)

    out = -bk.log10(bk.abs(det))
    # print(f, out)
    return out


{
    "gtol": 1e-6,
    "norm": "inf",
    "eps": 1e-6,
    "maxiter": None,
    "disp": False,
    "return_all": False,
    "finite_diff_rel_step": None,
}
#
# # jac = nn.grad(func)
# jac = None
#
# res = scipy.optimize.minimize(
#     func, x0, args=(), method="CG", jac=jac, tol=1e-6, callback=None, options=options
# )
#
# print(res.x)


def func(f):
    pw = nn.PlaneWave(wavelength=1 / f, angles=(0, 0, 0 * nn.pi / 2))
    sim = nn.Simulation([sup, ms, sub], pw, nh=nh, formulation=formulation)
    # S = sim.get_S_matrix()[0][1]
    # S = nn.helpers.block(S)
    # det = bk.linalg.det(S)
    # out = det#bk.abs(det)
    # print(f, out)
    R, T = sim.diffraction_efficiencies()
    return T


z0 = 0.1 - 0.1 * 1j
z1 = 0.53


freqs_re = np.linspace(z0.real, z1.real, 60)
freqs_im = np.linspace(z0.imag, z1.imag, 50)
#
plt.clf()
#
# DET = []
# for fre in freqs_re:
#     det_=[]
#     for fim in freqs_im:
#         f = fre + fim*1j
#         print(f)
#         det = func(f)
#
#         det_.append(det)
#
#         # plt.plot(f, bk.abs(det), "or")
#         # plt.pause(0.1)
#
#     DET.append(det_)
#
#
# DET = bk.array(DET).T
# plt.pcolor(freqs_re,freqs_im,bk.log10(abs(DET)))
# plt.plot(ev.real,ev.imag,"or")


tols = (1e-2 * (1 + 1j), 1e-2 * (1 + 1j), 1e-6 * (1 + 1j))
par_integ = (1e-4, 1e-4, 10)
tol_pol = 1e-6 * (1 + 1j)
tol_res = 1e-6 * (1 + 1j)
inv_golden_number = 2 / (1 + np.sqrt(5))
ratio = inv_golden_number
ratio_circ = 1 - inv_golden_number
nref_max = 100
ratio_re, ratio_im = ratio, ratio


poles_test = ev
#
#
# np.random.seed(200)
# Np = 10
# poles_test = ( 0.5*np.random.rand(Np) -
#               1j * 0.03*np.random.rand(Np))
# residues_test = ( np.random.rand(Np) + 1j *
#                  np.random.rand(Np)) * 1000
#
# isort = np.argsort(poles_test)
# poles_test = poles_test[isort]
# residues_test = residues_test[isort]
#
#
#
# def func(z):
#     N = len(poles_test)
#     out = 0
#     for i in range(N):
#         p = poles_test[i]
#         r = residues_test[i]
#         out += r / (z - p)
#     return out + z**2


# plt.close("all")

fig = plt.gcf()
ax = plt.gca()

# fig = plt.figure()
# ax = fig.add_subplot(111)  # , aspect='equal')
# ax.set_xlim((x0, x1))
# ax.set_ylim((y0, y1))
plt.xlabel(r"Re $z$")
plt.ylabel(r"Im $z$")
plt.gca().plot(np.real(poles_test), np.imag(poles_test), "sk")

poles, residues, nb_cuts = tc.pole_hunt(
    func,
    z0,
    z1,
    tols=tols,
    ratio_re=ratio_re,
    ratio_im=ratio_re,
    nref_max=1,
    ratio_circ=ratio_circ,
    tol_pol=tol_pol,
    tol_res=tol_res,
    par_integ=par_integ,
    poles=[],
    residues=[],
    nb_cuts=0,
)
# tc.pole_hunt(func, z0, z1)
