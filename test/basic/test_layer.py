#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

import numpy as npo
import pytest

npo.random.seed(84)
N = 3


def test_layer():
    import nannos as nn

    l = nn.Layer("test", 1)
    assert l.name == "test"
    assert l.thickness == 1
    assert l.__repr__() == "Layer test"

    matrix = npo.random.rand(N, N) + 1j * npo.random.rand(N, N)
    matrix = nn.backend.array(matrix)
    w, v = l.solve_eigenproblem(matrix)

    lc = l.copy()
    lc.name = "copy"
    lc.thickness = 2.1
    assert npo.allclose(lc.eigenvalues, w)
    assert npo.allclose(lc.eigenvectors, v)

    with pytest.raises(ValueError) as excinfo:
        l = nn.Layer("test", -0.1)
    assert "thickness must be positive." == str(excinfo.value)


#
#
# def test_eig():
#     i = 0
#     vals, vects = [], []
#     matrix_eig = npo.random.rand(N, N) + 1j * npo.random.rand(N, N)
#     for backend in ["autograd", "jax", "torch","scipy", "numpy"]:
#         import nannos as nn
#         nn.set_backend(backend)
#         matrix_eig_ = nn.backend.array(matrix_eig)
#         l = nn.Layer("test", 1)
#         w, v = l.solve_eigenproblem(matrix_eig_)
#         # v = v / v[0, 0]
#         vals.append(w)
#         vects.append(v)
#         if i > 0:
#             assert npo.allclose(w, vals[0])
#             assert npo.allclose(v, vects[0])
#         i += 1
#     # nn.set_backend("torch")
