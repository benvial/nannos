#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import numpy as np
import pytest

from nannos.formulations.tangent import get_tangent_field

n2 = 7
Nx, Ny = 2 ** n2, 2 ** n2
radius = 0.1
grid = np.ones((Nx, Ny), dtype=float)
x0 = np.linspace(0, 1.0, Nx)
y0 = np.linspace(0, 1.0, Ny)
x, y = np.meshgrid(x0, y0, indexing="ij")
hole = (x - 0.3) ** 2 + (y - 0.3) ** 2 < radius ** 2
square = np.logical_or(np.abs(x + 0.7) < 0.1, np.abs(y + 0.7) < 0.2)
square0 = np.logical_and(x > 0.6, x < 0.8)
square1 = np.logical_and(y > 0.5, y < 0.8)
square = np.logical_and(square1, square0)

grid[hole] = 0
grid[square] = 0


def test_tangent():
    t = get_tangent_field(grid)
    norm = np.linalg.norm(t, axis=0)
    assert np.allclose(norm, 1)
