#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


import numpy as npo
import pytest

import nannos as nn

pi = npo.pi

np = nn.backend

nh = 51
L1 = [1.0, 0]
L2 = [0, 1.0]
Nx = 2**5
Ny = 2**5
eps_pattern = 4.0
eps_hole = 1.0
mu_pattern = 1.0
mu_hole = 1.0
h = 2

lattice = nn.Lattice((L1, L2), discretization=Nx)
sup = lattice.Layer("Superstrate", epsilon=1, mu=1)
sub = lattice.Layer("Substrate", epsilon=1, mu=1)


def build_pattern(anisotropic=False):
    radius = 0.25
    hole = lattice.circle((0.5, 0.5), radius)

    ids = np.ones((Nx, Ny), dtype=np.complex128)
    zs = np.zeros_like(ids)

    if anisotropic:
        exx = np.where(hole, ids * eps_hole, ids * eps_pattern)
        eyy = np.where(hole, ids * eps_hole, ids * eps_pattern * 1)
        ezz = np.where(hole, ids * eps_hole, ids * eps_pattern * 2)
        epsgrid = np.array([[exx, zs, zs], [zs, eyy, zs], [zs, zs, ezz]])
        mxx = np.where(hole, ids * mu_hole, ids * mu_pattern)
        myy = np.where(hole, ids * mu_hole, ids * mu_pattern)
        mzz = np.where(hole, ids * mu_hole, ids * mu_pattern)
        mugrid = np.array([[mxx, zs, zs], [zs, mxx, zs], [zs, zs, mzz]])
    else:
        epsgrid = np.where(hole, ids * eps_hole, ids * eps_pattern)
        mugrid = np.where(hole, ids * mu_hole, ids * mu_pattern)
    return epsgrid, mugrid


@pytest.mark.parametrize("wl", [0.9, 1.1])
@pytest.mark.parametrize("theta", [0, 30])
@pytest.mark.parametrize("phi", [0, 30])
@pytest.mark.parametrize("psi", [0, 30])
def test_uniform(wl, theta, phi, psi):
    pw = nn.PlaneWave(wavelength=wl, angles=(theta, phi, psi))
    # eps = np.diag([2, 3, 4])
    # mu = np.diag([5, 6, 7])
    eps = 4
    mu = 1
    eps_sup, eps_sub = 1.0, 1.0
    mu_sup, mu_sub = 1, 1

    sup = lattice.Layer("Superstrate", epsilon=eps_sup, mu=mu_sup)
    sub = lattice.Layer("Substrate", epsilon=eps_sub, mu=mu_sub)

    un = lattice.Layer("Uniform", h, epsilon=eps, mu=mu)
    sim = nn.Simulation([sup, un, sub], pw, nh=5)
    R, T = sim.diffraction_efficiencies()
    B = R + T
    assert npo.allclose(B, 1)

    # sup = Layer("Superstrate", epsilon=mu_sup, mu=eps_sup)
    # sub = Layer("Substrate", epsilon=mu_sub, mu=eps_sub)
    #
    # un = Layer("Uniform", h, epsilon=mu, mu=eps)
    # sim = Simulation( [sup, un, sub], pw, nh)
    # Rdual, Tdual = sim.diffraction_efficiencies()
    # assert npo.allclose(Rdual + Tdual, 1)
    #
    # print(R)
    # print(Rdual)
    # print(T)
    # print(Tdual)
    #
    # assert npo.allclose(R, Rdual, rtol=1e-3)
    # assert npo.allclose(T, Tdual, rtol=1e-3)


def hole_array(epsgrid, mugrid, pw, nh=nh, formulation="original"):
    st = lattice.Layer("Structured", h)
    st.epsilon = epsgrid
    st.mu = mugrid
    sim = nn.Simulation([sup, st, sub], pw, nh, formulation=formulation)
    return sim


formulations = ["original", "tangent", "jones", "pol"]


@pytest.mark.parametrize("wl", [0.9, 1.1])
@pytest.mark.parametrize("theta", [0, 30])
@pytest.mark.parametrize("phi", [0, 30])
@pytest.mark.parametrize("psi", [0, 30])
@pytest.mark.parametrize("formulation", formulations)
def test_fft(wl, theta, phi, psi, formulation):
    pw = nn.PlaneWave(wavelength=wl, angles=(theta, phi, psi))

    epsgrid, mugrid = build_pattern(anisotropic=False)
    sim = hole_array(epsgrid, mugrid, pw, formulation=formulation)
    R, T = sim.diffraction_efficiencies()
    B = R + T

    print(">>> formulation = ", formulation)
    print("T = ", T)
    print("R = ", R)
    print("R + T = ", B)
    assert npo.allclose(B, 1, atol=1e-1)

    a, b = sim._get_amplitudes(1, z=0.1)
    field_fourier = sim.get_field_fourier(1, z=0.1)

    sim.get_field_grid(1)
    sim.get_z_stress_tensor_integral(1)

    # epsgrid, mugrid = build_pattern(eps, mu, anisotropic=True)
    # simu_aniso = hole_array(epsgrid, mugrid, pw, formulation=formulation)
    # Raniso, Taniso = simu_aniso.diffraction_efficiencies()
    # Baniso = Raniso + Taniso
    # # assert npo.allclose(Baniso, 1,atol=1e-3)
    #
    # print(">>> formulation (anisotropic)= ", formulation)
    # print("T = ", Taniso)
    # print("R = ", Raniso)
    # print("R + T = ", Baniso)
    # assert npo.allclose(Baniso, 1,atol=1e-1)
    # assert npo.allclose(R, Raniso)
    # assert npo.allclose(T, Taniso)
    return R, T, sim


# for f in formulations:
#
#     R,T,sim = test_fft(1.7, 0, 0, 0,f)
#     print("**********************************8")
#     #
#     # print(">>> formulation = ", f)
#     # print("T = ", (T))
#     # print("R = ", (R))
#     # B = (R + T)
#     # print("R + T = ", B)
# #
