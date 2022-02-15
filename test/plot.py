#!/usr/bin/env python


import numpy as np
from mayavi import mlab

import nannos as nn

L1 = [1.0, 0]
L2 = [0, 1.0]

Nx = 2**9
Ny = 2**9

eps_pattern = 4.0
eps_hole = 1

radius = 0.25
x0 = np.linspace(0, 1.0, Nx)
y0 = np.linspace(0, 1.0, Ny)
x, y = np.meshgrid(x0, y0, indexing="ij")
hole = (x - 0.5) ** 2 + (y - 0.5) ** 2 < radius**2

ids = np.ones((Nx, Ny), dtype=float)
epsgrid = ids * eps_pattern
epsgrid[hole] = eps_hole
mlab.barchart(x, y, 0 * y, epsgrid, extent=[0, 1, 0, 1, 0, 0.02], lateral_scale=1 / Nx)
mlab.show()
