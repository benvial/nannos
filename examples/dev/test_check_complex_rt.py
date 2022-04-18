#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import sys

import matplotlib.pyplot as plt
import numpy as np

import nannos as nn

plt.ion()
plt.close("all")

nh = 151
formulation = "original"

wl = 532
# wl = float(sys.argv[1])

wls = np.linspace(745, 750, 51)
wls = [748.5]
for wl in wls:
    P = 300
    H = 400
    # H = wl*0.5
    A, Phi = [], []
    lattice = nn.Lattice([[P, 0], [0, P]], discretization=2**9)
    sup = lattice.Layer("Superstrate", epsilon=1.0**2)
    sub = lattice.Layer("Substrate", epsilon=1.0**2)
    epsilon1 = (4 + 0.00 * 1j) ** 2
    epsilon = lattice.ones() * 1
    metaatom = lattice.circle((0.5 * P, 0.5 * P), 0.3 * P)
    epsilon[metaatom] = epsilon1
    ms = lattice.Layer("Metasurface", thickness=H, epsilon=epsilon)
    pw = nn.PlaneWave(wl, angles=(20, 0, 0))

    # pw = nn.PlaneWave(wavelength=1/1 / wl, angles=(0.3 * nn.pi / 2, 0.2, 1))
    sim = nn.Simulation([sup, ms, sub], pw, nh=nh, formulation=formulation)
    R, T = sim.diffraction_efficiencies()
    print(T, R, R + T)

    nin = (sim.layers[0].epsilon * sim.layers[0].mu) ** 0.5
    nout = (sim.layers[-1].epsilon * sim.layers[-1].mu) ** 0.5
    # norma = (nin/nout)**0.5

    norma_t = 1 / (nout * nin) ** 0.5

    aN = sim.get_field_fourier("Substrate")[0, 0, 0:3]
    tx = sim.get_order(aN[0], (0, 0)) / norma_t
    ty = sim.get_order(aN[1], (0, 0)) / norma_t
    tz = sim.get_order(aN[2], (0, 0)) / norma_t
    t = np.abs(tx) ** 2 + np.abs(ty) ** 2 + np.abs(tz) ** 2

    print("complex T")
    print(tx, ty, tz)

    T1 = np.abs(tx) ** 2 + np.abs(ty) ** 2 + np.abs(tz) ** 2

    norma_r = 1 / (nin * nin) ** 0.5
    b0 = sim.get_field_fourier("Superstrate", z=0)[0, 0, 0:3]
    rx = sim.get_order(b0[0], (0, 0)) / norma_r - sim.excitation.amplitude[0]
    ry = sim.get_order(b0[1], (0, 0)) / norma_r - sim.excitation.amplitude[1]
    rz = sim.get_order(b0[2], (0, 0)) / norma_r - sim.excitation.amplitude[2]

    print("complex R")
    print(rx, ry, rz)

    R1 = np.abs(rx) ** 2 + np.abs(ry) ** 2 + np.abs(rz) ** 2

    print(T1, R1, R1 + T1)

    # plt.plot(wl,R,"sb")
    # plt.plot(wl,T,"sr")
    # plt.pause(0.1)

# sys.exit(0)

# ai0, bi0 = sim._get_amplitudes("Substrate")
#
# layer = sup
# self = sim
# bk = nn.backend
#
#
# C = bk.linalg.inv(layer.Qeps) * (self.omega * layer.eigenvalues)
# et = bk.array([-pw.amplitude[1], pw.amplitude[0]])
# out = C @ et
#
#
# print(pw.amplitude[:2])
# # print(sim.a0)
#
# kt = pw.wavevector
#
# K = bk.array([[kt[1] ** 2, -kt[0] * kt[1]], [-kt[0] * kt[1], kt[0] ** 2]])
# Q = self.omega**2 * bk.eye(2) - K
# q = (self.omega**2 - kt[0] ** 2 - kt[1] ** 2) ** 0.5
# C = bk.linalg.inv(Q) * (self.omega * q)
# out = C @ et
#
#
# # print(out)
# #
# # layer = self.lattice.Layer()
# # epsilon = layer.epsilon
# # Keps = _build_Kmatrix(1 / epsilon * self.IdG, Ky, -Kx)
# # Pmu = block([[mu * self.IdG, self.ZeroG], [self.ZeroG, mu * self.IdG]])
# # # Qeps = self.omega**2 * Pmu - Keps
#
#
# rx = -0.5533+0.0215j
# ry = 0
# rz = -0.5533+0.0215j
# print("complex R")
# print(rx, ry, rz)
#
# R1 = (np.abs(rx) ** 2 + np.abs(ry) ** 2 + np.abs(rz) ** 2)
#
# print(T1, R1, R1 + T1)

#
#
# tx2 = -0.019-0.4787j
# ty2 = 0
# tz2 = 0.0139+0.3491j
# T2 = np.abs(tx2) ** 2 + np.abs(ty2) ** 2 + np.abs(tz2) ** 2
#
# print("complex T 2")
# print(tx2, ty2, tz2)
#
# print(T2)
#
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
