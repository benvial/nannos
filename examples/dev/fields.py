#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

import sys

import matplotlib.pyplot as plt
import numpy as np

import nannos as nn

nn.set_log_level("INFO")

plt.ion()
plt.close("all")

nh = 10**2
formulation = "original"
formulation = "tangent"

wl = 0.5

Lambda = 0.25

a = 0.5 * Lambda
h = Lambda
lattice = nn.Lattice([[Lambda, 0], [0, Lambda]], discretization=2**9)
sup = lattice.Layer("Superstrate", epsilon=1.0**2)
sub = lattice.Layer("Substrate", epsilon=1.45**2)
epsilon1 = (0.97 + 1.87j) ** 2
epsilon = lattice.ones() * 1
rect = lattice.rectangle((0.5 * Lambda + 0.25 * a, 0.5 * Lambda), (0.5 * a, a))
circ = lattice.circle((0.5 * Lambda, 0.5 * Lambda), 0.5 * a)

epsilon[rect] = epsilon1
epsilon[circ] = epsilon1
ms = lattice.Layer("Metasurface", thickness=h, epsilon=epsilon)

# ms.plot()

pw = nn.PlaneWave(wl, angles=(0, 0, 0))
sim = nn.Simulation([sup, ms, sub], pw, nh=nh, formulation=formulation)
R, T = sim.diffraction_efficiencies()
print(T, R, R + T)

# #
# E, H = sim.get_field_grid("Metasurface", z=h / 2)
# Enorm2 = np.abs(E[0])
# print(Enorm2.shape)
# plt.figure()
# plt.pcolormesh(*lattice.grid, Enorm2[:, :, 0], cmap="inferno")
# plt.colorbar()
# ms.plot(alpha=0.1, cmap="Greys")
# plt.axis("off")
# plt.tight_layout()
# plt.show()

#
# E, H = sim.get_field_grid("Metasurface",z=h/2)
# Enorm2 = np.abs(E[0])
# print(Enorm2.shape)
# plt.figure()
# plt.pcolormesh(*lattice.grid, Enorm2[:, :, 0], cmap="inferno")
# plt.colorbar()
# ms.plot(alpha=0.1, cmap="Greys")
# plt.axis("off")
# plt.tight_layout()
# plt.show()


ds = 1  # 2**2
grid = lattice.grid[:, ::ds, ::ds]
nplot = int(lattice.discretization[0] / ds)

Ex = sim.get_field_grid(
    "Metasurface",
    field="E",
    component="x",
    z=np.linspace(h / 2, h / 2, 1),
    shape=(nplot, nplot),
)

# Ex = sim.get_field_grid("Metasurface",field="E",component="x")
Enorm2 = np.abs(Ex)
print(Enorm2.shape)
plt.figure()
plt.pcolormesh(*grid, Enorm2[:, :, 0], cmap="inferno")
plt.colorbar()
ms.plot(alpha=0.1, cmap="Greys")
plt.axis("off")
plt.title("k space standard")
plt.tight_layout()
plt.show()

fields_fourier = sim.get_field_fourier(1, h / 2)
fe = fields_fourier[:, 0]
amplitudes = fe[:, 0, :]
f = 0
x, y = grid

amplitudes = np.array(amplitudes)
if len(amplitudes.shape) == 1:
    amplitudes = np.reshape(amplitudes, amplitudes.shape + (1,))

for i in range(sim.nh):
    expo = np.exp(1j * (sim.kx[i] * x + sim.ky[i] * y))
    f += amplitudes[:, i] * expo

Enorm2 = np.abs(f)
print(Enorm2.shape)
plt.figure()
plt.pcolormesh(*grid, Enorm2, cmap="inferno")
plt.colorbar()
ms.plot(alpha=0.1, cmap="Greys")
plt.axis("off")
plt.title("real space standard")
plt.tight_layout()
plt.show()
#
# test = np.abs(Ex[:, :, 0]/f)
#
# test = sim.get_epsilon(ms).real
#
#
# plt.figure()
# plt.pcolormesh(*grid, test, cmap="inferno")
# plt.colorbar()
# ms.plot(alpha=0.1, cmap="Greys")
# plt.axis("off")
# plt.title("k space standard")
# plt.tight_layout()
# plt.show()
#
#
# xs
#
#
# layer = ms
# self = sim
# t = ms.get_tangent_field(self.harmonics, normalize=False)
# t1 = np.array(t).real
#
# plt.figure()
# plt.quiver(*grid[:,::10,::10],*t1[:,::10,::10])
# plt.axis("scaled")


################ smooth fields #################
layer = ms
self = sim
from nannos.utils import block, get_block

Z = np.linspace(0.5 * h, h, 1)
fields_fourier = sim.get_field_fourier(1, Z)
fe = fields_fourier[:, 0]

bk = nn.backend
t = layer.get_tangent_field(self.harmonics, normalize=False)
T = block([[t[1], bk.conj(t[0])], [-t[0], bk.conj(t[1])]])
# N = int(T.shape[0]/2)
N, nuhat_inv = self._get_nu_hat_inv(layer)
# That = block(
#     [
#         [self._get_toeplitz_matrix(get_block(T, i, j, N)) for j in range(2)]
#         for i in range(2)
#     ]
# )
# Nvec = 1 - T
Nvec = block([[1 - t[1], bk.conj(t[0])], [-t[0], 1 - bk.conj(t[1])]])
Nhat = block(
    [
        [self._get_toeplitz_matrix(get_block(Nvec, i, j, N)) for j in range(2)]
        for i in range(2)
    ]
)

if layer.is_epsilon_anisotropic:
    eps_para_hat = self._get_toeplitz_matrix(epsilon, transverse=True)
else:
    eps_para_hat = [[nuhat_inv, self.ZeroG], [self.ZeroG, nuhat_inv]]
Peps = block(eps_para_hat)


# shape=(nplot, nplot)
shape = layer.epsilon.shape
fields = bk.zeros((len(Z), 2, 3, self.nh), dtype=bk.complex128)

for iz, z_ in enumerate(Z):
    et = bk.hstack([-fe[iz, 1], fe[iz, 0]])
    dNt = 0.5 * (Nhat @ Peps + Peps @ Nhat) @ et
    dNx = dNt[self.nh :]
    fields[iz, 0, 0] = dNx
# # epsi1 = np.broadcast_to(layer.epsilon,(7,512,512)).T
# epsi1 =layer.epsilon
# dNx = np.array([dNx])
# ex = np.array([fe[iz, 0]])
# ex = fe[iz, 0,:]
# _Ex1 = self.get_ifft_amplitudes(dNx,shape)[0]/epsi1
# _Ex2 = self.get_ifft_amplitudes(ex,shape)[0]
# Ex[iz] = _Ex2


dNx = fields[:, 0, 0]

Ex1 = self.get_ifft_amplitudes(dNx, shape)
Ex2 = self.get_ifft_amplitudes(fe[:, 0, :], shape)
epsi1 = bk.reshape(layer.epsilon, layer.epsilon.shape + (1,))
Ex1 /= epsi1
Ex = Ex1 + Ex2
# Enorm2 = np.abs(Ex[:,:,0])
# Enorm2 = np.abs(Ex1[0,:,:])
#
# Ex=Ex1[0,:,:] + Ex2[:,:,0]
Enorm2 = np.abs(Ex[:, :, 0])
# Enorm2 = np.abs(Ex2[:,:,0])

# Enorm2 = np.abs( Ex2[:,:,0])
print(Enorm2.shape)
plt.figure()
grid = lattice.grid
plt.pcolormesh(*grid, Enorm2, cmap="inferno")
plt.colorbar()
ms.plot(alpha=0.1, cmap="Greys")
plt.axis("off")
plt.title("k space tangent")
plt.tight_layout()
plt.show()
#
# plt.pcolormesh(*grid, epsi1[:,:,0].real, cmap="inferno")
#

# # nx,ny,nz = 2*sim.nh,2*sim.nh, 100
# nx, ny, nz = sim.nh, sim.nh * 2, 200
#
# x = np.linspace(0, P, nx)
# y = np.linspace(0, P, ny)
# i = 1
# layer = sim.layers[i]
# t = layer.thickness
# z = np.linspace(0, t, nz)
# # E, H = sim.get_field_grid(i,z,shape=(nx,ny))
# bk = nn.backend
# self = sim
# fields_fourier = self.get_field_fourier(1, z)
# fe = fields_fourier[:, 0]
# fh = fields_fourier[:, 1]
# amplitudes = fe[:, 0, :]
#
# shape = (nx, ny)
#
#
# amplitudes = bk.array(amplitudes)
# if len(amplitudes.shape) == 1:
#     amplitudes = bk.reshape(amplitudes, amplitudes.shape + (1,))
# s = 0
# for i in range(self.nh):
#     print(i)
#     f = bk.zeros(shape + (amplitudes.shape[0],), dtype=bk.complex128)
#     try:
#         f[self.harmonics[0, i], self.harmonics[1, i]] = 1.0
#         # f = np.repeat(f,amplitudes.shape[0],axis=-1)
#         a = amplitudes[:, i]
#         s += a * f
#     except:
#         pass
#
# a = nn.formulations.fft.inverse_fourier_transform(s, axes=(0, 1))
# plt.clf()
#
# # plt.pcolormesh(z,x,a[:,int(ny/2),:].real,cmap="RdBu_r")
# plt.pcolormesh(z, y, a[int(nx / 2), :, :].real, cmap="RdBu_r")
# plt.axis("scaled")
# plt.colorbar()
# plt.tight_layout()
# sys.exit(0)
#
#
# i = 0
#
#
# Z = []
# Es = []
# Hs = []
# z0 = 0
# for i, layer in enumerate(sim.layers):
#     t = wl if layer.thickness == 0 else layer.thickness
#
#     if i == 0:
#         z = np.linspace(-t, 0, nz)
#     else:
#         z = np.linspace(0, t, nz)[1:]
#
#     E, H = sim.get_field_grid(i, z, shape=(nx, ny))
#     # if i>0:
#     #     z = z[1:]
#     #     E = E[:,:,:,1:]
#     #     H = H[:,:,:,1:]
#     Z.append(z0 + z)
#     Es.append(E)
#     Hs.append(H)
#     z0 += z[-1]
#
#
# E = np.concatenate(Es, axis=-1)
# H = np.concatenate(Hs, axis=-1)
# z = np.concatenate(Z, axis=-1)
#
#
# # z=np.linspace(-400,0,nz)
# x = np.linspace(0, P, nx)
# y = np.linspace(0, P, ny)
# #
# # E, H = sim.get_field_grid(i,z,shape=(nx,ny))
#
#
# plt.clf()
# # x1,z1 = np.meshgrid(x,z)
#
#
# case = "y"
#
# plt.close("all")
# for F in [E, H]:
#     fig, ax = plt.subplots(3, 2)
#     for i in range(3):
#         for j in range(2):
#             f = F[i][:, int(ny / 2), :] if case == "x" else F[i][int(nx / 2), :, :]
#             c = x if case == "x" else y
#             f = np.real(f) if j == 0 else np.imag(f)
#             im = ax[i][j].pcolormesh(z, c, f, cmap="RdBu_r")
#             plt.colorbar(im, ax=ax[i][j])
#             ax[i][j].set_aspect(1)
#             plt.tight_layout()
#
# fig, ax = plt.subplots(1, 2)
# for i, F in enumerate([E, H]):
#     n = 0
#     for j in range(3):
#         f = F[j][:, int(ny / 2), :]
#         n = np.abs(f) ** 2
#     im = ax[i].pcolormesh(z, x, n**0.5, cmap="inferno")
#     plt.colorbar(im, ax=ax[i])
#     ax[i].set_aspect(1)
#     plt.tight_layout()
#
#
# # amplitudes1 =fe[:, 2, :]
# # shape = (nx,ny)
# #
# # #
# # # amplitudes = bk.array(amplitudes)
# # # if len(amplitudes.shape) == 1:
# # #     amplitudes = bk.reshape(amplitudes, amplitudes.shape + (1,))
# #
# # from nannos.utils import set_index
# # # # self.get_ifft_amplitudes(amplitudes, shape)
# # # for i in range(self.nh):
# # #     print(self.harmonics[0, i], self.harmonics[1, i])
# # #
# #
# #
# # amplitudes = bk.array(amplitudes)
# # if len(amplitudes.shape) == 1:
# #     amplitudes = bk.reshape(amplitudes, amplitudes.shape + (1,))
# # s = 0
# # s1 = 0
# # for i in range(self.nh):
# #     print(i)
# #     f = bk.zeros(shape + (amplitudes.shape[0],), dtype=bk.complex128)
# #     try:
# #         set_index(f, [self.harmonics[0, i], self.harmonics[1, i]], 1.0)
# #         # f[self.harmonics[0, i], self.harmonics[1, i], :] = 1.0
# #         # f = bk.zeros(shape)
# #         f[self.harmonics[0, i], self.harmonics[1, i]] = 1.0
# #         # f = np.repeat(f,amplitudes.shape[0],axis=-1)
# #         a = amplitudes[:, i]
# #         s += a * f
# #         a = amplitudes1[:, i]
# #         s1 += a * f
# #     except:
# #         pass
# #
# # ft = nn.formulations.fft.inverse_fourier_transform(s, axes=(0, 1))
# # ft1 = nn.formulations.fft.inverse_fourier_transform(s1, axes=(0, 1))
# # plt.clf()
# # x1,z1 = np.meshgrid(x,z)
# # plt.pcolormesh(z,x,ft1[:,int(ny/2),:].real,cmap="RdBu_r")
# # plt.axis("scaled")
# # plt.colorbar()
# # plt.tight_layout()
