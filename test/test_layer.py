#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

import numpy as np
import pytest

from nannos.layers import Layer

np.random.seed(84)
N = 3
matrix_eig = np.random.rand(N, N) + 1j * np.random.rand(N, N)


def test_layer():
    l = Layer("test", 1)
    assert l.name == "test"
    assert l.thickness == 1
    assert l.__repr__() == "Layer test"

    matrix = np.random.rand(N, N) + 1j * np.random.rand(N, N)
    w, v = l.solve_eigenproblem(matrix)

    lc = l.copy()
    lc.name = "copy"
    lc.thickness = 2.1
    assert np.allclose(lc.eigenvalues, w)
    assert np.allclose(lc.eigenvectors, v)

    with pytest.raises(ValueError) as excinfo:
        l = Layer("test", -0.1)
    assert "thickness must be positive." == str(excinfo.value)


def test_eig():
    i = 0
    vals, vects = [], []
    for backend in ["autograd", "jax", "torch", "numpy"]:
        import nannos as nn

        nn.set_backend(backend)
        l = nn.Layer("test", 1)
        w, v = l.solve_eigenproblem(matrix_eig)
        vals.append(w)
        vects.append(v)
        if i > 0:
            assert np.allclose(w, vals[0])
            # assert np.allclose(v,vects[0])
        i += 1
