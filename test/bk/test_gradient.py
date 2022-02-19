#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

import random

import pytest

random.seed(1984)
Nx = Ny = 16

xlist = [random.random() for _ in range(Nx * Ny)]

formulations = ["original"]


@pytest.mark.parametrize("formulation", formulations)
def test_grad(formulation):
    res = dict()
    dres = dict()
    import nannos as nn

    backends = ["autograd"]
    if nn.HAS_TORCH:
        backends.append("torch")
    for backend in backends:

        nn.set_backend(backend)
        from nannos import grad
        from nannos import numpy as np

        def f(var):
            xa = var.reshape(Nx, Ny)
            eps_pattern = 9 + 1 * xa
            sup = nn.Layer("Superstrate")
            sub = nn.Layer("Substrate")
            ms = nn.Layer("ms", 1)
            pattern = nn.Pattern(eps_pattern)
            ms.add_pattern(pattern)
            sim = nn.Simulation(
                nn.Lattice(([1, 0], [0, 1])),
                [sup, ms, sub],
                nn.PlaneWave(1.1),
                10,
                formulation=formulation,
            )
            R, T = sim.diffraction_efficiencies()

            return R

        x = nn.backend.array(xlist, dtype=nn.backend.float64)
        y = f(x)
        res[backend] = y

        def first_finite_differences(f, x):
            eps = 1e-2
            return nn.backend.array(
                [
                    (f(x + eps * v) - f(x - eps * v)) / (2 * eps)
                    for v in nn.backend.eye(len(x))
                ],
            )

        dy = nn.grad(f)(x)
        dres[backend] = dy

        dy_fd = first_finite_differences(f, x)
        err = nn.backend.linalg.norm(dy - dy_fd)
        print(y)
        print(err)

        assert nn.backend.allclose(dy, dy_fd, atol=1e-7)

    # nn.set_backend("numpy")
