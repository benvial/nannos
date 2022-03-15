#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


"""
Tangent field
=============


"""


plt.ion()
plt.close("all")

#############################################################################
# We will generate a field tangent to the material interface

bk = npg  # nn.backend

nh = 100

n2 = 11
n2_down = 6
Nx, Ny = 2 ** n2, 2 ** n2
radius = 0.25
grid = bk.ones((Nx, Ny), dtype=float) * 5
x0 = bk.linspace(0, 1.0, Nx)
y0 = bk.linspace(0, 1.0, Ny)
x, y = bk.meshgrid(x0, y0, indexing="ij")
hole = (x - 0.3) ** 2 + (y - 0.3) ** 2 < radius ** 2
square0 = bk.logical_and(x > 0.7, x < 0.9)
square1 = bk.logical_and(y > 0.2, y < 0.8)
square = bk.logical_and(square1, square0)

grid[hole] = 1
grid[square] = 1

t = nn.formulations.tangent.get_tangent_field(grid, normalize=True)
#
# plt.imshow(grid,cmap="tab20c",extent=(0,1,0,1),origin="lower")
# dsp = 10
# plt.quiver(
#     x[::dsp, ::dsp], y[::dsp, ::dsp], t[0][::dsp, ::dsp], t[1][::dsp, ::dsp], scale=44
# )
# plt.axis("scaled")
# # _=plt.axis("off")
# plt.show()


# plt.imshow(grid,cmap="tab20c",origin="lower")
# dsp = 10
# plt.quiver(
#     x[::dsp, ::dsp]*Nx, y[::dsp, ::dsp]*Nx, t[0][::dsp, ::dsp], t[1][::dsp, ::dsp], scale=44
# )
# # plt.axis("scaled")
# _=plt.axis("off")
# plt.show()
#


st = nn.Layer("pat", 1)
st.add_pattern(nn.Pattern(grid))
lays = [nn.Layer("sup"), st, nn.Layer("sub")]
pw = nn.PlaneWave(1.2)
self = nn.Simulation(nn.Lattice(((1, 0), (0, 1))), lays, pw, nh=nh)

# coef = npg.random.rand(self.nh)+1j*npg.random.rand(self.nh)
# coef = (npg.random.rand(self.nh) - 0.5) * 2 + 1j * (npg.random.rand(self.nh) - 0.5) * 2
# coef[:10] = 0
# a = self.get_ifft_amplitudes([coef], shape=(Nx, Ny))[:, :, 0]

nh_new = self.nh

self.nh = nh_new

self.harmonics = self.harmonics[:, : self.nh]


def _normalize(x, n):
    with npg.errstate(invalid="ignore"):
        f = x / (n)
    return bk.array(bk.where(n == 0.0, 0.0 * x, f))


def get_amps(amplitudes, shape=(Nx, Ny)):
    amplitudes = bk.array([amplitudes])
    if len(amplitudes.shape) == 1:
        amplitudes = bk.reshape(amplitudes, amplitudes.shape + (1,))
    f = bk.zeros(shape + (amplitudes.shape[0],) + (self.nh,), dtype=bk.complex128)
    f[self.harmonics[0], self.harmonics[1], 0, :] = bk.eye(self.nh)
    s = bk.sum(amplitudes * f, axis=-1)
    ft = npg.fft.ifft2(s, axes=(0, 1)) * Nx * Ny
    return ft[:, :, 0]


def get_toeplitz_matrix(u):
    uft = npg.fft.fft2(u, axes=(0, 1)) / (Nx * Ny)
    ix = bk.arange(self.nh)
    jx, jy = bk.meshgrid(ix, ix, indexing="ij")
    delta = self.harmonics[:, jx] - self.harmonics[:, jy]
    return uft[delta[0, :], delta[1, :]]


# a = get_amps(coef)


xf = apply_filter(grid, rfilt=6)
dgrid_f = npg.array(npg.gradient(xf))
# dgrid_f = npg.array(npg.gradient(grid))
# dgrid_f /= norm(dgrid_f)
maxi = bk.max(norm(dgrid_f))
aa = bk.linalg.norm(dgrid_f, axis=0) > maxi / 2

normdgrid_f = norm(dgrid_f)
# dgrid_f = npg.array([_normalize(dgrid_f[i], normdgrid_f) for i in range(2)])
# dgrid_f /= norm(dgrid_f)

dgrid_f = npg.array([dgrid_f[i] / bk.max(normdgrid_f) for i in range(2)])

fig, ax = plt.subplots(1, 2)
ax[0].imshow(dgrid_f[0].real, origin="lower")
ax[1].imshow(dgrid_f[1].real, origin="lower")
# plt.imshow(a.real)


downsample = 2 ** (n2 - n2_down)
shape_small = (int(Nx / downsample), int(Ny / downsample))  # (int(aa[aa].shape[0]/2),2)
shape_small = (downsample, downsample)  # (int(aa[aa].shape[0]/2),2)


#
# v = bk.gradient(grid)
# vf = bk.gradient(xf)

coef0x = get_toeplitz_matrix(dgrid_f[0])
coef0y = get_toeplitz_matrix(dgrid_f[1])
# print(minifun_der(x0))
#


fx = bk.fft.fftfreq(Nx)
fy = bk.fft.fftfreq(Ny)

Fx, Fy = bk.meshgrid(fx, fy, indexing="ij")

# Fx1 = get_toeplitz_matrix(Fx)
# Fy1 = get_toeplitz_matrix(Fy)

ix = bk.arange(self.nh)
jx, jy = bk.meshgrid(ix, ix, indexing="ij")
delta = self.harmonics[:, jx] - self.harmonics[:, jy]
Fx1 = Fx[delta[0, :], delta[1, :]]
Fy1 = Fy[delta[0, :], delta[1, :]]

# Fx1 = bk.fft.fftshift(Fx1)
# Fy1 = bk.fft.fftshift(Fy1)


coefx = get_toeplitz_matrix(xf) * Fx1 * 1j * 2 * nn.pi
coefy = get_toeplitz_matrix(xf) * Fy1 * 1j * 2 * nn.pi

#
# plt.figure()
# plt.imshow(coef0y.real)
# plt.colorbar()
# plt.figure()
# plt.imshow(coefy.real)
# plt.colorbar()
#
#
# plt.figure()
# plt.imshow(coef0y.imag)
# plt.colorbar()
# plt.figure()
# plt.imshow(coefy.imag)
# plt.colorbar()

# assert bk.allclose(coef0x,coefx)
# assert bk.allclose(coef0y,coefy)

# import sys
# sys.exit(0)


def minifun(x):
    coef = x[: self.nh] + 1j * x[self.nh :]
    a = get_amps(coef, shape_small)
    da = npg.array(npg.gradient(a))
    # I = npg.mean(npg.abs(da[:, aa] - dgrid_f[:, aa]) ** 2)
    I = npg.mean(
        npg.abs(da - dgrid_f[:, :: int(Nx / downsample), :: int(Ny / downsample)]) ** 2
    )
    print(I)
    return I


def minifun1(x):
    coef = x[: self.nh] + 1j * x[self.nh :]
    c0x = npg.array(coef0x)
    c0y = npg.array(coef0y)
    # c1 = npg.array(coef)
    c1x = npg.array(coef) * Fx1 * 1j * 2 * nn.pi
    c1y = npg.array(coef) * Fy1 * 1j * 2 * nn.pi
    # I = npg.mean(npg.abs(1-c1x/c0x) ** 2) + npg.mean(npg.abs(1-c1y/c0y) ** 2)
    I = npg.sum(npg.abs(c1x - c0x) ** 2) + npg.sum(npg.abs(c1y - c0y) ** 2)
    print(I)
    return I


minifun_der = grad(minifun)

x0 = npg.zeros(2 * self.nh)
# x0 = npg.random.rand(2 * self.nh)

# a0 = get_amps(xf)
# x0 = a0.real.tolist()[0] + a0.imag.tolist()[0]


res = minimize(
    minifun,
    x0,
    method="BFGS",
    jac=minifun_der,
    options={"gtol": 1e-3, "disp": True, "maxiter": 20},
)


xopt = res.x
coef = xopt[: self.nh] + 1j * xopt[self.nh :]

a = get_amps(coef)
# plt.imshow(a.real)
t = npg.array(npg.gradient(a))
# t = npg.array([get_amps(coef0x),get_amps(coef0y)])
t = [t[1], -t[0]]

# t = dgrid_f
#
norm_t = norm(t)
# norm_t = 1
#
t = [_normalize(t[i], norm_t) for i in range(2)]


# t = [t[i] / bk.max(norm(t)) for i in range(2)]

# t = [t[i] / norm_t for i in range(2)]
# t = [t[1],-t[0]]
plt.figure()
# plt.pcolor(x, y, grid, cmap="tab20c")
plt.imshow(grid.T, cmap="tab20c", origin="lower", extent=(0, 1, 0, 1))
dsp = int(Nx / downsample)
plt.quiver(
    x[::dsp, ::dsp],
    y[::dsp, ::dsp],
    t[0][::dsp, ::dsp],
    t[1][::dsp, ::dsp],
    scale=20,
)
plt.axis("scaled")
_ = plt.axis("off")
plt.show()
