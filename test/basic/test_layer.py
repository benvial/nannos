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

    l = nn.layers.Layer("test", 1)
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
        l = nn.layers.Layer("test", -0.1)
    assert "thickness must be positive." == str(excinfo.value)
