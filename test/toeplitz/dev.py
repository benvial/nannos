#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


import sys
import scipy
import matplotlib.pyplot as plt
import numpy as np

import nannos as nn

plt.close("all")
plt.ion()

form = "original"
# form = "tangent"
nh = int(sys.argv[1])
theta = 30
psi = 11
phi = 22
pw = nn.PlaneWave(wavelength=1, angles=(theta, phi, psi))


#################  1D

print(" ---------  1D  ---------------")
lattice = nn.Lattice(1, 2**10)  # ,truncation="parallelogrammic")
# harmonics_array0 = lattice.get_harmonics(nh,sort=False)[0]
# harmonics_array = lattice.get_harmonics(nh,sort=True)[0]
# lattice = nn.Lattice(1, 2**10,harmonics_array=harmonics_array)
eps_metal = (3-1j*1e-9) ** 2
epsgrid = lattice.ones() * 5
stripe = lattice.stripe(0.5, 0.5)
epsgrid[stripe] = eps_metal
sup = lattice.Layer("Superstrate")
sub = lattice.Layer("Substrate", epsilon=eps_metal)
grating = lattice.Layer("Grating", thickness=1)
grating.epsilon = epsgrid
stack = [sup, grating, sub]
sim = nn.Simulation(stack, pw, nh, formulation=form)
harmonics0 = sim.harmonics.copy()
eps_hat = sim._get_toeplitz_matrix(epsgrid)
# srt = np.argsort(sim.harmonics[0])
# sim.harmonics = sim.harmonics[:, srt]
# sim.harmonics = harmonics_array
# print(harmonics0)
# print(sim.harmonics)

# Ri,Ti = sim.diffraction_efficiencies(orders=True)
# print(Ri)
# print(Ti)
# R,T = sim.diffraction_efficiencies()
# print(R,T,R+T)

# sys.exit(0)

eps_hat_toeplitz = sim._get_toeplitz_matrix(epsgrid)


# sys.exit(0)

b = np.eye(len(eps_hat))
# b = np.ones(len(eps_hat))
c = eps_hat_toeplitz[:, 0]  # First column of T
r = eps_hat_toeplitz[0, :]  # First row of T


ttoeplitz = nn.tic()
eps_hat_inv_toeplitz = scipy.linalg.solve_toeplitz((c, r), b, check_finite=False)
ttoeplitz = nn.toc(ttoeplitz)
t1 = nn.tic()
eps_hat_inv = scipy.linalg.solve(eps_hat_toeplitz, b, check_finite=False)
# eps_hat_inv = np.linalg.inv(eps_hat_toeplitz)
t1 = nn.toc(t1)
assert np.allclose(eps_hat_inv, eps_hat_inv_toeplitz)

# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(eps_hat_toeplitz.real)
# ax[0].set_title("unsorted")
# ax[1].imshow(eps_hat_inv_toeplitz.real)
# ax[1].set_title("sorted")


print("nh = ", nh)
print("shape = ", eps_hat_toeplitz.shape)
print("speedup = ", t1 / ttoeplitz)

sim.build_matrix(grating)
M = grating.matrix
t = nn.tic()
evs, modes = grating.solve_eigenproblem(M)
# evs = evs**2
t = nn.toc(t)

# sim.harmonics[0] = np.sort(sim.harmonics[0])
# sim.harmonics[1,0] = 1
# print(sim.harmonics)


fig, ax = plt.subplots(1, 2)
ax[0].imshow(M.real)
ax[0].set_title("real")
ax[1].imshow(M.imag)
ax[1].set_title("imag")

plt.figure()

# evs /= np.prod(M.shape)
plt.plot(evs.real, evs.imag, "o", c="k")
# plt.ylim(-0.02,0.02)
# plt.xlim(-3,1)

M00 = nn.get_block(M, 0, 0, sim.nh)
M11 = nn.get_block(M, 1, 1, sim.nh)
t1 = nn.tic()
evs0, modes0 = grating.solve_eigenproblem(M00)
evs1, modes1 = grating.solve_eigenproblem(M11)
# evs0 = evs0**2
# evs1 = evs1**2
t1 = nn.toc(t1)
print("speedup block= ", t / t1)
plt.plot(evs1.real, evs1.imag, "xr")
plt.plot(evs0.real, evs0.imag, "+b")

evsblock = np.hstack([evs0,evs1])
srt = np.argsort(abs(evsblock))
evsblock = evsblock[srt]
srt = np.argsort(abs(evs))
evs = evs[srt]
assert np.allclose(evs,evsblock)




# for i in range(2):
#     for j in range(2):
#         _M = nn.get_block(M, i, j, sim.nh)
#         plt.figure()
#         plt.imshow(abs(_M))
#         plt.title(f"{i},{j}")


sys.exit(0)


grating2 = lattice.Layer("Grating", thickness=1)
grating2.epsilon = epsgrid
stack = [sup, grating2, sub]
sim = nn.Simulation(stack, pw, nh, formulation=form)
sim.harmonics = harmonics0
sim.build_matrix(grating2)
M2 = grating2.matrix
t2 = nn.tic()
evs2, modes2 = grating2.solve_eigenproblem(M2)

t2 = nn.toc(t2)
# assert np.allclose(evs,evs2)
plt.plot(evs2.real, evs2.imag, "sg")


sys.exit(0)


# M = M11
# sparse = scipy.sparse.coo_array(M)
# density = sparse.getnnz() / np.prod(M.shape)
# print(density)
# print(len(evs))

# sys.exit(0)

# # sim = nn.Simulation(stack, pw, 4*nh, formulation=form)
# # sim.build_matrix(grating)
# # M = grating.matrix
# # sparse = scipy.sparse.coo_array(M)

# neig = int(sim.nh * density)
# t1 = nn.tic()
# evs_sparse, modes_sparse = scipy.sparse.linalg.eigs(sparse, k=neig, sigma=None)
# t1 = nn.toc(t1)
# print(len(evs_sparse))
# print("speedup eigenvalues sparse = ", t / t1)

# # evs_sparse = evs_sparse**0.5
# # evs_sparse = np.where(np.imag(evs_sparse) < 0.0, -evs_sparse, evs_sparse)

# # plt.figure()
# plt.plot(evs.real, evs.imag, "o", c="k")
# plt.plot(evs_sparse.real, evs_sparse.imag, "+", c="r")
# plt.ylim(-3740.0876323792363, 2806.967282257585)
# plt.xlim(-1595714.216946054, 89660.1861679831)

# # plt.figure()
# # plt.imshow(abs(M))
# # sys.exit(0)

# # P = nn.get_block(M, 0, 0, nh)
# # plt.imshow(P.real)

# # sys.exit(0)


#################   2D

print(" ---------  2D  ---------------")


lattice = nn.Lattice(
    [[1.0, 0], [0, 1.0]], discretization=(2**9, 2**9), truncation="parallelogrammic"
)


sup = lattice.Layer("Superstrate", epsilon=1)
ms = lattice.Layer("Metasurface", thickness=0.5)
sub = lattice.Layer("Substrate", epsilon=1)

ms.epsilon = lattice.ones() * 12.0
circ = lattice.circle(center=(0.5, 0.5), radius=0.2)
ms.epsilon[circ] = 1

stack = [sup, ms, sub]
sim = nn.Simulation(stack, pw, nh=nh)
grating = ms

sim.build_matrix(grating)
M = grating.matrix
t = nn.tic()
evs, modes = grating.solve_eigenproblem(M)
# evs = evs**2
t = nn.toc(t)


# plt.clf()

# M = nn.get_block(M, 0,1, sim.nh)
# M -= np.diag(M)
# plt.imshow(np.abs(M))
# plt.colorbar()

plt.figure()

# evs /= np.prod(M.shape)
plt.plot(evs.real, evs.imag, "o", c="k")
# plt.ylim(-0.02,0.02)
# plt.xlim(-3,1)

M00 = nn.get_block(M, 0, 0, sim.nh)
M11 = nn.get_block(M, 1, 1, sim.nh)
t1 = nn.tic()
evs0, modes0 = grating.solve_eigenproblem(M00)
evs1, modes1 = grating.solve_eigenproblem(M11)
t1 = nn.toc(t1)
print("speedup block= ", t / t1)
plt.plot(evs1.real, evs1.imag, "xr")
plt.plot(evs0.real, evs0.imag, "+b")

evsblock = np.hstack([evs0,evs1])
evs = np.sort(evs)
evsblock = np.sort(evsblock)
assert np.allclose(evs,evsblock)


# fig, ax = plt.subplots(1, 2)
# # sim.harmonics[0] = np.sort(sim.harmonics[0])
# # sim.harmonics[1,0] = 1
# print(sim.harmonics)
# eps_hat = sim._get_toeplitz_matrix(ms.epsilon)
# ax[0].imshow(eps_hat.real)
# ax[0].set_title("unsorted")

# # sim.harmonics = np.sort(sim.harmonics)
# # sim.harmonics = sim.harmonics[srt]
# # # srt = np.argsort(np.linalg.norm(sim.harmonics, axis=0))
# srt = np.argsort(sim.harmonics[0])
# sim.harmonics = sim.harmonics[:, srt]
# srt = np.argsort(sim.harmonics[1])
# sim.harmonics = sim.harmonics[:, srt]
# # srt = np.argsort(np.sum(sim.harmonics,axis=0))
# # sim.harmonics = sim.harmonics[:, srt]

# # # N = 4
# # sim.harmonics = np.array(
# #     [[-1, -1, -1, 0, 0, 0, 1, 1, 1], [0, -1, 1, 0, -1, 1, 0, -1, 1]]
# # )
# # sim.harmonics = np.array(
# #      [[-1, 0, 1, -1, 0, 1,-1, 0, 1],[-1, -1, -1, 0, 0, 0, 1, 1, 1]]
# # )

# # sim.harmonics = np.zeros((2, (2 * N + 1) ** 2))
# # for i in range(-N, N + 1):
# #     for j in range(-N, N + 1):
# #         sim.harmonics[:,i+j] = 1
# print(sim.harmonics)
# eps_hat = sim._get_toeplitz_matrix(ms.epsilon)
# ax[1].imshow(eps_hat.real)
# ax[1].set_title("sorted")
