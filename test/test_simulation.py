#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import numpy as np
import pytest

from nannos import Lattice, Layer, Pattern, PlaneWave, Simulation, pi

nG = 51
L1 = [1.0, 0]
L2 = [0, 1.0]
freq = 1.1
theta = 30.0 * pi / 180
phi = 30.0 * pi / 180
psi = 30.0 * pi / 180


pw = PlaneWave(
    frequency=freq, angles=(theta * pi / 180, phi * pi / 180, psi * pi / 180)
)

Nx = 2 ** 9
Ny = 2 ** 9

eps_pattern = 4.0
eps_hole = 1.0
mu_pattern = 1.0
mu_hole = 1.0

h = 2

radius = 0.25
x0 = np.linspace(0, 1.0, Nx)
y0 = np.linspace(0, 1.0, Ny)
x, y = np.meshgrid(x0, y0, indexing="ij")
hole = (x - 0.5) ** 2 + (y - 0.5) ** 2 < radius ** 2

lattice = Lattice((L1, L2))
sup = Layer("Superstrate", epsilon=1, mu=1)
sub = Layer("Substrate", epsilon=1, mu=1)


ids = np.ones((Nx, Ny), dtype=float)
zs = np.zeros_like(ids)


eps = eps_pattern, eps_hole
mu = mu_pattern, mu_hole


def build_pattern(eps, mu, anisotropic=False):
    eps_pattern, eps_hole = eps
    mu_pattern, mu_hole = mu

    if anisotropic:
        exx = ids * eps_pattern
        eyy = ids * eps_pattern * 3
        ezz = ids * eps_pattern * 1
        exx[hole] = eps_hole
        eyy[hole] = eps_hole
        ezz[hole] = eps_hole
        epsgrid = np.array([[exx, zs, zs], [zs, eyy, zs], [zs, zs, ezz]])

        mxx = ids * mu_pattern
        myy = ids * mu_pattern
        mzz = ids * mu_pattern
        mxx[hole] = mu_hole
        myy[hole] = mu_hole
        mzz[hole] = mu_hole
        mugrid = np.array([[mxx, zs, zs], [zs, mxx, zs], [zs, zs, mzz]])
    else:
        epsgrid = ids * eps_pattern
        epsgrid[hole] = eps_hole
        mugrid = ids * mu_pattern
        mugrid[hole] = mu_hole

    return epsgrid, mugrid


@pytest.mark.parametrize("freq", [0.7, 1.1])
@pytest.mark.parametrize("theta", [0, 30])
@pytest.mark.parametrize("phi", [0, 30])
@pytest.mark.parametrize("psi", [0, 30])
def test_uniform(freq, theta, phi, psi):
    pw = PlaneWave(
        frequency=freq, angles=(theta * pi / 180, phi * pi / 180, psi * pi / 180)
    )
    # eps = np.diag([2, 3, 4])
    # mu = np.diag([5, 6, 7])
    eps = 4
    mu = 1
    eps_sup, eps_sub = 1.0, 1.0
    mu_sup, mu_sub = 1, 1

    sup = Layer("Superstrate", epsilon=eps_sup, mu=mu_sup)
    sub = Layer("Substrate", epsilon=eps_sub, mu=mu_sub)

    un = Layer("Uniform", h, epsilon=eps, mu=mu)
    simu = Simulation(lattice, [sup, un, sub], pw, nG=5)
    R, T = simu.diffraction_efficiencies()
    B = R + T
    assert np.allclose(B, 1)

    # sup = Layer("Superstrate", epsilon=mu_sup, mu=eps_sup)
    # sub = Layer("Substrate", epsilon=mu_sub, mu=eps_sub)
    #
    # un = Layer("Uniform", h, epsilon=mu, mu=eps)
    # simu = Simulation(lattice, [sup, un, sub], pw, nG)
    # Rdual, Tdual = simu.diffraction_efficiencies()
    # assert np.allclose(Rdual + Tdual, 1)
    #
    # print(R)
    # print(Rdual)
    # print(T)
    # print(Tdual)
    #
    # assert np.allclose(R, Rdual, rtol=1e-3)
    # assert np.allclose(T, Tdual, rtol=1e-3)


def hole_array(epsgrid, mugrid, pw, nG=nG, formulation="original"):
    pattern = Pattern(epsgrid, mugrid)
    st = Layer("Structured", h)
    st.add_pattern(pattern)
    simu = Simulation(lattice, [sup, st, sub], pw, nG, formulation=formulation)
    return simu


formulations = ["original", "normal", "jones", "pol"]


@pytest.mark.parametrize("freq", [0.7, 1.1])
@pytest.mark.parametrize("theta", [0, 30])
@pytest.mark.parametrize("phi", [0, 30])
@pytest.mark.parametrize("psi", [0, 30])
@pytest.mark.parametrize("formulation", formulations)
def test_fft(freq, theta, phi, psi, formulation):
    pw = PlaneWave(
        frequency=freq, angles=(theta * pi / 180, phi * pi / 180, psi * pi / 180)
    )

    epsgrid, mugrid = build_pattern(eps, mu, anisotropic=False)
    simu = hole_array(epsgrid, mugrid, pw, formulation=formulation)
    R, T = simu.diffraction_efficiencies()
    B = R + T

    print(">>> formulation = ", formulation)
    print("T = ", T)
    print("R = ", R)
    print("R + T = ", B)
    assert np.allclose(B, 1, atol=1e-1)

    a, b = simu._get_amplitudes(1, z=0.1)
    field_fourier = simu.get_field_fourier(1, z=0.1)

    # epsgrid, mugrid = build_pattern(eps, mu, anisotropic=True)
    # simu_aniso = hole_array(epsgrid, mugrid, pw, formulation=formulation)
    # Raniso, Taniso = simu_aniso.diffraction_efficiencies()
    # Baniso = Raniso + Taniso
    # # assert np.allclose(Baniso, 1,atol=1e-3)
    #
    # print(">>> formulation (anisotropic)= ", formulation)
    # print("T = ", Taniso)
    # print("R = ", Raniso)
    # print("R + T = ", Baniso)
    # assert np.allclose(Baniso, 1,atol=1e-1)
    # assert np.allclose(R, Raniso)
    # assert np.allclose(T, Taniso)
    return R, T, simu


# for f in formulations:
#
#     R,T,simu = test_fft(1.7, 0, 0, 0,f)
#     print("**********************************8")
#     #
#     # print(">>> formulation = ", f)
#     # print("T = ", (T))
#     # print("R = ", (R))
#     # B = (R + T)
#     # print("R + T = ", B)
# #
