#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import numpy as np

from nannos import Lattice, Layer, Pattern, PlaneWave, Simulation, pi

formulations = ["original", "normal", "jones", "pol"]
nG = 101
L1 = [1.0, 0]
L2 = [0, 1.0]
freq = 1.6
theta = 30.0 * pi / 180
phi = 0.0 * pi / 180
psi = 0.0 * pi / 180


pw = PlaneWave(frequency=freq, angles=(theta, phi, psi))

Nx = 2 ** 8
Ny = 2 ** 8

eps_pattern = 4.0
eps_hole = 1.0
mu_pattern = 1.0
mu_hole = 1.0

h = 1  # 0.5
l = h  # 1/freq

hsup = hsub = h  # 2*l

radius = 0.25
x0 = np.linspace(0, 1.0, Nx)
y0 = np.linspace(0, 1.0, Ny)
x, y = np.meshgrid(x0, y0, indexing="ij")
hole = (x - 0.5) ** 2 + (y - 0.5) ** 2 < radius ** 2

lattice = Lattice((L1, L2))
sup = Layer("Superstrate", epsilon=1, mu=1, thickness=hsup)
sub = Layer("Substrate", epsilon=1, mu=1, thickness=hsub)


ids = np.ones((Nx, Ny), dtype=float)
zs = np.zeros_like(ids)

epsgrid = ids * eps_pattern
epsgrid[hole] = eps_hole
mugrid = ids * mu_pattern
mugrid[hole] = mu_hole


formulation = "original"
# formulation="normal"
pattern = Pattern(epsgrid, mugrid)
st = Layer("Structured", h)
st.add_pattern(pattern)
simu = Simulation(lattice, [sup, st, sub], pw, nG, formulation=formulation)

thicknesses = [l.thickness for l in simu.layers]


R, T = simu.diffraction_efficiencies()
B = R + T


print(">>> formulation = ", formulation)
print("T = ", T)
print("R = ", R)
print("R + T = ", B)
assert np.allclose(B, 1, atol=1e-1)


# Ri, Ti = simu.diffraction_efficiencies(orders=True)
# nmax=4
# print("Ti = ", Ti[:nmax])
# print("Ri = ", Ri[:nmax])
# print("R + T = ", B)


import matplotlib.pyplot as plt

plt.ion()
plt.close("all")
layer_index = 1

z0 = 0

RE = []
Z = []
for layer_index in range(3):

    z = 0

    nz = 100
    z = np.linspace(0, thicknesses[layer_index], nz)
    shape = 2 ** 6, 2 ** 6
    E, H = simu.get_field_grid(layer_index, z=z, shape=shape)

    #
    # plt.figure()
    # plt.plot(z, field_fourier[:, 1, 1, :].T.real)

    # (ex,ey,ez),(hx,hy,hz) = f
    x0 = np.linspace(0, 1, shape[0])
    y0 = np.linspace(0, 1, shape[1])
    x1, y1 = np.meshgrid(x0, y0)
    x = np.broadcast_to(x1, (nz, shape[0], shape[1])).T
    y = np.broadcast_to(y1, (nz, shape[0], shape[1])).T
    kr = simu.k0para[0] * x + simu.k0para[1] * y
    ex = E[0] * np.exp(1j * kr)
    # ex = np.exp(1j*kr)

    p = np.abs(E[0]) ** 2 + np.abs(E[1]) ** 2 + np.abs(E[2]) ** 2

    p = ex

    # p = np.exp(1j*3*x)

    q = np.exp(1j * (simu.k0para[0] * x1 + simu.k0para[1] * y1))

    iz = 0

    test = p[:, :, iz]
    # plt.figure()
    # plt.imshow(test.real, extent=(0, L1[0], 0, L2[1]), cmap="RdBu_r", origin="lower")
    # # plt.imshow(test.real, extent=(0, L1[0], 0, L2[1]), cmap="inferno", origin="lower")
    # plt.colorbar()
    # plt.imshow(epsgrid, extent=(0, L1[0], 0, L2[1]), cmap="Greys", origin="lower",alpha=0.2)

    # p = ex[0, :, :].T* np.exp(1j * simu.k0para[0] * x)

    a = ex.T  # *q
    re = a[:, 0, :].real

    # re = np.flipud(re)

    # re = np.abs(ex[0, :, :].T)

    # plt.figure()
    dz = z[1] - z[0]
    z2 = z + z0
    # plt.imshow(re, extent=(0, L1[0], z0, z0+z[-1]+dz), cmap="RdBu_r", origin="lower")
    # plt.pcolor(x0,z2,re, cmap="RdBu_r",shading='auto')
    z0 += thicknesses[layer_index]
    # plt.colorbar()
    # plt.title("z")

    if layer_index > 0:
        a = re[1:, :]
        b = z2[1:]
    else:
        a = re
        b = z2
    RE.append(a)
    Z.append(b)

Z = np.hstack(Z)
RE = np.vstack(RE)
plt.pcolor(x0, Z, RE, cmap="RdBu_r", shading="flat")

plt.ylim([0, sum(thicknesses)])
plt.axis("scaled")

xsx
plt.clf()

x = np.linspace(0, 1, shape[0])

# x, z = np.meshgrid(x, z)


plt.figure()

plt.plot(z, p[0, :].real)
plt.plot(z, p[20, :].real, "ok")

xs
#
# plt.figure()
# plt.imshow(kr[1,:,:].real)
# plt.colorbar()

# from nannos.formulations import fft
# shape = 50,50
# self = simu
# amplitudes = field_fourier[0][0][0]
#
# s=0
# for i in range(self.nG):
#     f = np.zeros(shape,dtype=complex)
#     f[self.G[0,i], self.G[1,i]] = 1.
#     s +=  amplitudes[i]*f
# p = fft.inverse_fourier_transform(s)
#
#
# G = self.G.T
# s_in = amplitudes
# Nx,Ny = shape
# dN = 1./Nx/Ny
# nG,_ = G.shape
#
# s0 = 0#np.zeros((Nx,Ny),dtype=complex)
# for i in range(nG):
#     x = G[i,0]
#     y = G[i,1]
#
#     stmp = np.zeros((Nx,Ny),dtype=complex)
#     stmp[G[i,0],G[i,1]] = 1.
#     s0 += s_in[i]*stmp
#
# ez = fft.inverse_fourier_transform(s0)
#
#


plt.clf()
plt.imshow(p.T.real, extent=(0, L1[0], 0, h), cmap="RdBu_r")
plt.colorbar()

xsxs

q = np.array(field_fourier[0])

feh = field_fourier[0]
fe = np.array(feh[0])


u = np.fft.ifft2(fe, s=(100, 100))

print(u.shape)

ex, ey, ez = np.fft.ifft2(fe, s=(100, 100))

ex, ey, ez = fft.inverse_fourier_transform(fe, shape)

f = simu.get_field_grid(1, z=0.1, shape=(100, 100))


q = f[1][1]
print(q.shape)
#
# f =simu.get_field_grid(1, z=0.1)
# q = f[1][1]
# print(q.shape)

b = np.random.rand(30, 30)


u = np.fft.ifft2(b)
print(u.shape)

u = np.fft.ifft2(b, s=(100, 100))
print(u.shape)


b = np.random.rand(3, 30, 30)


u = np.fft.ifft2(b)
print(u.shape)

u = np.fft.ifft2(b, s=(100, 100))
print(u.shape)


from nannos.formulations import fft


def get_ifft(amplitudes, shape, G):
    """
    Reconstruct real-space fields
    """
    nG = G.shape[1]

    s = 0
    for i in range(nG):
        f = np.zeros(shape, dtype=complex)
        f[G[:, i]] = 1.0
        s += amplitudes[i] * f
    return fft.inverse_fourier_transform(s0)
