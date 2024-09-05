#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: GPLv3


import modal

devices = ["cpu", "gpu"]
nfreq = 2


gpu = "any"
# gpu = "A10G"
# gpu = "H100"

app = modal.App(f"nannos benchmarks")
vol = modal.Volume.from_name("nannos-volume", create_if_missing=True)
nannos_image = modal.Image.from_dockerfile("Dockerfile", gpu=gpu)


@app.function(image=nannos_image, volumes={"/data": vol}, gpu=gpu, timeout=3600)
def test_nannos(param):
    backend, device = param
    import numpy as npo

    import nannos as nn

    print("--------------------------")
    print(f"{backend} {device}")
    print("--------------------------")
    nn.set_backend(backend)
    if device == "gpu":
        nn.use_gpu(True)
    else:
        nn.use_gpu(False)
    if nn.BACKEND == "torch":
        nn.backend.device(nn.DEVICE)

    L1 = [1.0, 0]
    L2 = [0, 1.0]
    Nx = 2**9
    Ny = 2**9
    formulation = "original"
    eps_pattern = 4.0 + 0j
    h = 2
    radius = 0.25
    x0 = nn.backend.linspace(0, 1.0, Nx)
    y0 = nn.backend.linspace(0, 1.0, Ny)
    x, y = nn.backend.meshgrid(x0, y0, indexing="ij")
    hole = (x - 0.5) ** 2 + (y - 0.5) ** 2 < radius**2
    hole = nn.backend.array(hole)

    lattice = nn.Lattice((L1, L2))
    sup = lattice.Layer("Superstrate", epsilon=1, mu=1)
    sub = lattice.Layer("Substrate", epsilon=1, mu=1)

    eps1 = nn.backend.array(eps_pattern, dtype=nn.backend.complex128)
    eps2 = nn.backend.array(1, dtype=nn.backend.complex128)
    epsgrid = nn.backend.where(hole, eps1, eps2)
    # epsgrid[hole] = eps_pattern

    st = lattice.Layer("Structured", h)
    st.epsilon = epsgrid

    frequencies = nn.backend.ones(nfreq) * 1.1

    NH = [100, 200, 400, 600, 800, 1000]
    NH_real = []
    TIMES = []
    TIMES_ALL = []

    for nh in NH:
        print(f"number of harmonics = {nh}")

        TIMES_NH = []
        for ifreq, freq in enumerate(frequencies):
            pw = nn.PlaneWave(
                wavelength=1 / freq,
            )
            t0 = nn.tic()
            sim = nn.Simulation(
                [sup, st, sub],
                pw,
                nh,
                formulation=formulation,
            )
            R, T = sim.diffraction_efficiencies()
            t1 = nn.toc(t0)
            print(t1)
            if ifreq > 0:
                TIMES_NH.append(t1)

        TIMES.append(sum(TIMES_NH) / len(TIMES_NH))
        TIMES_ALL.append(TIMES_NH)

        NH_real.append(sim.nh)

    npo.savez(
        f"/data/benchmark_{backend}_{device}.npz",
        times=TIMES,
        times_all=TIMES_ALL,
        real_nh=NH_real,
        nh=NH,
    )

    vol.commit()  # Needed


@app.local_entrypoint()
def main():
    backends = ["numpy", "scipy", "autograd", "jax", "torch"]
    inputs = [(b, "cpu") for b in backends]
    inputs += [(b, "gpu") for b in ["jax", "torch"]]
    for param in test_nannos.map(inputs):
        pass


# to download the output locally run:
# modal volume get nannos-volume testnannos.npz

# import os

# os.system(f"modal volume get nannos-volume {output_name}")
