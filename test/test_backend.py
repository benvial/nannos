#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

import pytest


def test_backend():
    import nannos as nn

    assert nn.get_backend() == "numpy"
    assert nn.BACKEND == "numpy"

    nn.set_backend("scipy")
    assert nn.numpy.__name__ == "numpy"
    assert nn.backend.__name__ == "numpy"
    assert nn.get_backend() == "scipy"
    assert nn.BACKEND == "scipy"

    nn.set_backend("autograd")
    assert nn.numpy.__name__ == "autograd.numpy"
    assert nn.backend.__name__ == "autograd.numpy"
    assert nn.get_backend() == "autograd"
    assert nn.BACKEND == "autograd"

    nn.set_backend("jax")
    assert nn.numpy.__name__ == "jax.numpy"
    assert nn.backend.__name__ == "jax.numpy"
    assert nn.get_backend() == "jax"
    assert nn.BACKEND == "jax"

    nn.set_backend("torch")
    assert nn.numpy.__name__ == "numpy"
    if nn.has_torch():
        assert nn.get_backend() == "torch"
        assert nn.backend.__name__ == "torch"
        assert nn.BACKEND == "torch"

    with pytest.raises(ValueError) as excinfo:
        nn.set_backend("fake")
    assert "Unknown backend" in str(excinfo.value)
    nn.set_backend("numpy")
    assert nn.numpy.__name__ == "numpy"
    assert nn.backend.__name__ == "numpy"
    assert nn.get_backend() == "numpy"
    assert nn.BACKEND == "numpy"


formulations = ["original", "tangent", "jones"]
backends = ["numpy", "scipy", "autograd", "jax", "torch"]


@pytest.mark.parametrize("formulation", formulations)
@pytest.mark.parametrize("backend", backends)
def test_simulations(formulation, backend):

    import nannos as nn

    nn.set_backend(backend)

    nh = 51
    L1 = [1.0, 0]
    L2 = [0, 1.0]
    Nx = 2 ** 9
    Ny = 2 ** 9

    eps_pattern = 4.0 + 0j
    eps_hole = 1.0
    mu_pattern = 1.0
    mu_hole = 1.0

    h = 2

    radius = 0.25
    x0 = nn.backend.linspace(0, 1.0, Nx)
    y0 = nn.backend.linspace(0, 1.0, Ny)
    x, y = nn.backend.meshgrid(x0, y0, indexing="ij")
    hole = (x - 0.5) ** 2 + (y - 0.5) ** 2 < radius ** 2

    lattice = nn.Lattice((L1, L2))
    sup = nn.Layer("Superstrate", epsilon=1, mu=1)
    sub = nn.Layer("Substrate", epsilon=1, mu=1)

    ids = nn.backend.ones((Nx, Ny), dtype=float)
    zs = nn.backend.zeros_like(ids)

    # epsgrid = ids * eps_pattern
    eps1 = nn.backend.array(eps_pattern, dtype=nn.backend.complex128)
    eps2 = nn.backend.array(1, dtype=nn.backend.complex128)
    epsgrid = nn.backend.where(hole, eps1, eps2)
    mugrid = 1 + 0j

    pattern = nn.Pattern(epsgrid, mugrid)
    st = nn.Layer("Structured", h)
    st.add_pattern(pattern)

    pw = nn.PlaneWave(
        frequency=1.1,
    )

    for i in range(1):
        t0 = nn.tic()
        sim = nn.Simulation(lattice, [sup, st, sub], pw, nh, formulation=formulation)
        R, T = sim.diffraction_efficiencies()
        nn.toc(t0)
    B = R + T

    print(">>> formulation = ", formulation)
    print("T = ", T)
    print("R = ", R)
    print("R + T = ", B)
    assert nn.backend.allclose(
        B, nn.backend.array(1.0, dtype=nn.backend.float64), atol=5e-3
    )

    a, b = sim._get_amplitudes(1, z=0.1)
    field_fourier = sim.get_field_fourier(1, z=0.1)

    nn.set_backend("numpy")
    return R, T, sim
