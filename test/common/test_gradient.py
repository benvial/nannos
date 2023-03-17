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
no_grad_backends = ["numpy", "scipy", "jax"]


@pytest.mark.parametrize("formulation", formulations)
def test_grad(formulation):
    res = {}
    dres = {}
    import nannos as nn

    backends = no_grad_backends + ["autograd"]

    if nn.HAS_TORCH:
        backends.append("torch")
    for backend in backends:
        nn.set_backend(backend)
        from nannos import grad
        from nannos import numpy as np

        def f(var):
            lattice = nn.Lattice(([1, 0], [0, 1]))
            xa = var.reshape(Nx, Ny)
            eps_pattern = 9 + 1 * xa
            sup = lattice.Layer("Superstrate")
            sub = lattice.Layer("Substrate")
            ms = lattice.Layer("ms", 1)
            ms.epsilon = eps_pattern
            sim = nn.Simulation(
                [sup, ms, sub],
                nn.PlaneWave(0.9),
                10,
                formulation=formulation,
            )
            R, T = sim.diffraction_efficiencies()

            return R

        if backend in no_grad_backends:
            with pytest.raises(NotImplementedError) as excinfo:
                nn.grad(f)
            assert "grad is not implemented" in str(excinfo.value)

        else:
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
