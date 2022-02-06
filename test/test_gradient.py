#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

import random

import pytest

random.seed(1984)
Nx = Ny = 11


# @pytest.mark.skip(reason="jax does not work")
def test_grad():
    # if True:
    res = dict()
    dres = dict()

    random.seed(84)
    x = [random.random() for _ in range(Nx * Ny)]
    # for backend in ["torch"]:
    for backend in ["autograd", "torch"]:
        import nannos as nn

        nn.set_backend(backend)
        from nannos import grad
        from nannos import numpy as np

        def f(x):
            xa = x.reshape(Nx, Ny)
            eps_pattern = 2 + 1 * xa
            sup = nn.Layer("Superstrate")
            sub = nn.Layer("Substrate")
            ms = nn.Layer("ms", 1)
            pattern = nn.Pattern(eps_pattern)
            ms.add_pattern(pattern)
            sim = nn.Simulation(
                nn.Lattice(([1, 0], [0, 1])), [sup, ms, sub], nn.PlaneWave(1.1), 6
            )
            # return eps_pattern.sum()
            # return pattern.epsilon.sum()
            # return sim.layers[1].patterns[0].epsilon.sum()
            R, T = sim.diffraction_efficiencies()
            return R

        x = nn.backend.array(x, dtype=nn.backend.float64)
        # x = np.array(x)
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

        assert nn.backend.allclose(dy, dy_fd)  # ,rtol=1e-8)

    nn.set_backend("numpy")

    # print(res["jax"], res["autograd"])
    # assert np.allclose(res["jax"], res["autograd"])
    # assert np.allclose(dres["jax"], dres["autograd"])
    # nn.set_backend("numpy")


# _x = x.clone().detach().requires_grad_(True)
# v = (_x)**2
# # v = v.clone().detach().requires_grad_(a.requires_grad)
# v = nn.backend.array(v)
# v = v.sum()
# nn.backend.autograd.grad(v, _x, allow_unused=True)


#
# import nannos as nn
#
# nn.set_backend("torch")
# import torch
#
# bk = torch
# _array = torch.tensor
#
# x = [random.random() for _ in range(Nx * Ny)]
#
# x = torch.tensor(x,requires_grad=True)
# xa =x.reshape(Nx, Ny)
# eps_pattern = 2 + 1 * xa**2
# sup = nn.Layer("Superstrate")
# sub = nn.Layer("Substrate")
# ms = nn.Layer("ms", 1)
# pattern = nn.Pattern(eps_pattern)
# ms.add_pattern(pattern)
# layers=[sup, ms, sub]
# excitation=nn.PlaneWave(1.1)
#
#
# k0para = _array(excitation.wavevector[:2]
#     * (layers[0].epsilon * layers[0].mu) ** 0.5
# )
#
#
# res = dict()
# dres = dict()
#
# random.seed(84)
# x = [random.random() for _ in range(Nx * Ny)]
# # for backend in ["jax", "autograd"]:
#
#
# for backend in ["torch"]:
#     import nannos as nn
#
#     nn.set_backend(backend)
#     from nannos import grad
#     from nannos import torch
#     from nannos import numpy as np
#
#     grad = torch.autograd.grad
#
#
#     x = torch.tensor(x,requires_grad=True)
#     # x.detach().numpy()
#
#
#     def f(x):
#
#         xa =x.reshape(Nx, Ny)
#         eps_pattern = 2 + 1 * xa**2
#         sup = nn.Layer("Superstrate")
#         sub = nn.Layer("Substrate")
#         ms = nn.Layer("ms", 1)
#         pattern = nn.Pattern(eps_pattern)
#         ms.add_pattern(pattern)
#         sim = nn.Simulation(
#             nn.Lattice(([1, 0], [0, 1])), [sup, ms, sub], nn.PlaneWave(1.1), 6
#         )
#         R, T = sim.diffraction_efficiencies()
#         return R
#
#
#     y = f(x)#eps_pattern.sum()
#
#
#     dy = grad(y,x)
#
#
#     res[backend] = y
#
#     # dy = grad(f)(xa)
#     dres[backend] = dy
#
#     def first_finite_differences(f, x):
#         eps = 1e-3
#         return np.array(
#             [(f(x + eps * v) - f(x - eps * v)) / (2 * eps) for v in np.eye(len(x))]
#         )
#
#     dy_fd = first_finite_differences(f, x)
#     assert np.allclose(dy, dy_fd)
