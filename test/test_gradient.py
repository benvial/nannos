#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

import random

import pytest

random.seed(1984)
Nx = Ny = 4


# @pytest.mark.skip(reason="jax does not work")
def test_grad():
    res = dict()
    dres = dict()

    random.seed(84)
    x = [random.random() for _ in range(Nx * Ny)]
    # for backend in ["jax", "autograd"]:
    for backend in ["autograd"]:
        import nannos as nn

        nn.set_backend(backend)
        from nannos import grad
        from nannos import numpy as np

        def f(x):
            xa = np.reshape(x, (Nx, Ny))
            eps_pattern = 2 + 1 * xa
            sup = nn.Layer("Superstrate")
            sub = nn.Layer("Substrate")
            ms = nn.Layer("ms", 1)
            pattern = nn.Pattern(eps_pattern)
            ms.add_pattern(pattern)
            sim = nn.Simulation(
                nn.Lattice(([1, 0], [0, 1])), [sup, ms, sub], nn.PlaneWave(1.1), 6
            )
            R, T = sim.diffraction_efficiencies()
            return R

        xa = np.array(x)
        y = f(xa)
        res[backend] = y

        dy = grad(f)(xa)
        dres[backend] = dy

        def first_finite_differences(f, x):
            eps = 1e-3
            return np.array(
                [(f(x + eps * v) - f(x - eps * v)) / (2 * eps) for v in np.eye(len(x))]
            )

        dy_fd = first_finite_differences(f, x)
        assert np.allclose(dy, dy_fd)
    # print(res["jax"], res["autograd"])
    # assert np.allclose(res["jax"], res["autograd"])
    # assert np.allclose(dres["jax"], dres["autograd"])
    # nn.set_backend("numpy")
