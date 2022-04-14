#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import numpy as npo

import nannos as nn

npo.random.seed(1984)


def simu(nn, nh):
    # nh = 100
    L1 = [1.0, 0]
    L2 = [0, 1.0]
    freq = 1.4
    bk = nn.backend
    theta = 30.0
    phi = 0.0
    psi = 0.0
    Nx = 2**9
    Ny = 2**9
    eps_sup = 1.0
    eps_pattern = 12.0
    eps_hole = 1.0
    eps_sub = 1.0
    h = 0.5
    radius = 0.2
    epsgrid = bk.ones((Nx, Ny), dtype=float) * eps_pattern
    x0 = bk.linspace(0, 1.0, Nx)
    y0 = bk.linspace(0, 1.0, Ny)
    x, y = bk.meshgrid(x0, y0, indexing="ij")
    hole = (x - 0.5) ** 2 + (y - 0.5) ** 2 < radius**2
    epsgrid[hole] = eps_hole
    lattice = nn.Lattice((L1, L2))

    sup = nn.Layer("Superstrate", epsilon=eps_sup)
    ms = nn.Layer("Metasurface", thickness=h)
    sub = nn.Layer("Substrate", epsilon=eps_sub)
    pattern = nn.Pattern(epsgrid, name="hole")
    ms.add_pattern(pattern)

    pw = nn.PlaneWave(wavelength=1 / freq, angles=(theta, phi, psi))
    stack = [sup, ms, sub]
    sim = nn.Simulation(stack, pw, nh)

    sim.solve()

    sim.get_field_fourier(1)

    # R, T = sim.diffraction_efficiencies()


def benchmark(f, N, nbatch):
    print("=================================")
    print(f"=============== {f} =============")
    print("=================================")
    M0 = npo.random.rand(nbatch, N, N) + 1j * npo.random.rand(nbatch, N, N)
    P0 = npo.random.rand(nbatch, N, N) + 1j * npo.random.rand(nbatch, N, N)
    # M0 = M0 + npo.conj(npo.transpose(M0, axes=(0,2,1)))

    res = dict()

    for backend in ["numpy", "torch"]:
        nn.set_backend(backend)
        for dev in ["cpu", "gpu"]:
            if dev == "gpu" and backend == "numpy":
                pass
            else:
                if dev == "gpu":
                    nn.use_gpu()

                if f == "fft":
                    F = nn.formulations.fft.fourier_transform
                elif f == "eig":
                    F = nn.backend.linalg.eig
                    # F = nn.backend.linalg.eigh

                elif f == "inv":
                    F = nn.backend.linalg.inv
                elif f == "matmul":
                    # F = lambda x: nn.backend.matmul(x[0],x[1])
                    F = nn.backend.matmul

                    # F =lambda M,P: M@P

                elif f == "simu":
                    F = lambda M: simu(nn, M.shape[-1])

                else:
                    raise ValueError

                M = nn.backend.array(M0)
                P = nn.backend.array(P0)

                if dev == "gpu":
                    # dummy evaluation to get rid of gpu loadiing time
                    I = F(M, P) if f == "matmul" else F(M)

                print(">>> batch")
                t = nn.tic()
                I = F(M, P) if f == "matmul" else F(M)
                res[backend + dev] = I
                tbatch = nn.toc(t)
                print(">>> loop")
                t = nn.tic()
                for i in range(nbatch):
                    I = F(M[i], P[i]) if f == "matmul" else F(M[i])
                tloop = nn.toc(t)

                print("--------------------------")

                print(f"speedup loop = {tloop/tbatch}")

                if backend != "numpy":
                    print(f"speedup numpy loop = {tloop_numpy/tloop}")
                    print(f"speedup numpy batch = {tbatch_numpy/tbatch}")
                else:
                    tloop_numpy = tloop
                    tbatch_numpy = tbatch

                if dev != "gpu":
                    tloop_torch = tloop
                    tbatch_torch = tbatch
                else:
                    print(f"speedup torch gpu loop = {tloop_torch/tloop}")
                    print(f"speedup torch gpu batch = {tbatch_torch/tbatch}")

    if f == "eig":
        ev = res["numpycpu"][0]
        ev = npo.sort(ev)
        ev1 = res["torchcpu"][0]
        ev1 = npo.sort(ev1)
        assert npo.allclose(ev1, ev)
        ev2 = res["torchgpu"][0].cpu()
        ev2 = npo.sort(ev2)
        assert npo.allclose(ev1, ev2)


# benchmark("fft", 2 ** 9, 10)
# benchmark("eig", 500, 10)
# benchmark("inv", 500, 10)

# benchmark("matmul", 800, 1)

benchmark("simu", 200, 1)
