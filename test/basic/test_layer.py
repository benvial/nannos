#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

import numpy as npo
import pytest

from nannos.utils import allclose

npo.random.seed(84)
N = 3


def test_layer():
    import nannos as nn

    lay = nn.layers.Layer("test", 1)
    assert lay.name == "test"
    assert lay.thickness == 1
    assert lay.__repr__() == "test"

    matrix = npo.random.rand(N, N) + 1j * npo.random.rand(N, N)
    matrix = nn.backend.array(matrix)
    w, v = lay.solve_eigenproblem(matrix)

    lc = lay.copy()
    lc.name = "copy"
    lc.thickness = 2.1
    assert allclose(lc.eigenvalues, w)
    assert allclose(lc.eigenvectors, v)

    with pytest.raises(ValueError) as excinfo:
        lay = nn.layers.Layer("test", -0.1)
    assert "thickness must be positive." == str(excinfo.value)
