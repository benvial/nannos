#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


import pytest


def test_tangent():
    import numpy as npo

    import nannos as nn

    bk = nn.backend

    n2 = 9
    Nx, Ny = 2 ** n2, 2 ** n2
    radius = 0.2
    grid = bk.ones((Nx, Ny), dtype=float)
    x0 = bk.linspace(0, 1.0, Nx)
    y0 = bk.linspace(0, 1.0, Ny)
    x, y = bk.meshgrid(x0, y0, indexing="xy")
    hole = (x - 0.3) ** 2 + (y - 0.3) ** 2 < radius ** 2
    square0 = bk.logical_and(x > 0.7, x < 0.9)
    square1 = bk.logical_and(y > 0.2, y < 0.8)
    square = bk.logical_and(square1, square0)
    obj = bk.logical_or(hole, square)
    grid = bk.where(obj, 0, 1)
    nh = 9
    harmonics = npo.arange(-nh, nh + 1)
    harmonicsx, harmonicsx = npo.meshgrid(harmonics, harmonics)
    harmonics = npo.vstack([harmonicsx.ravel(), harmonicsx.ravel()])
    t = nn.formulations.tangent.get_tangent_field(grid, harmonics, normalize=True)
    ta = [npo.array(t[i]) for i in range(2)]
    normt = npo.linalg.norm(npo.array(ta), axis=0)
    assert npo.allclose(normt, 1, atol=1e-6)

    t = nn.formulations.tangent.get_tangent_field(
        grid, harmonics, normalize=True, type="opt"
    )
    ta = [npo.array(t[i]) for i in range(2)]
    normt = npo.linalg.norm(npo.array(ta), axis=0)
    assert npo.allclose(normt, 1, atol=1e-3)

    with pytest.raises(ValueError) as excinfo:
        t = nn.formulations.tangent.get_tangent_field(None, None, type="fake")
    assert "Wrong type of tangent field" in str(excinfo.value)
