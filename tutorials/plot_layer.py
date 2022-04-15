#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

"""
Plotting layers
===============

Vizualizing a layer pattern in 2D with matplotlib.
"""


# sphinx_gallery_thumbnail_number = -1

import matplotlib.pyplot as plt

import nannos as nn

##############################################################################
# Let's define our lattice:


lat = nn.Lattice(basis_vectors=[[1, 0], [0, 1]], discretization=2**9)
epsilon = 6 * lat.ones()
hole = lat.circle(center=(0.5, 0.5), radius=0.2)
epsilon[hole] = 1

lay = lat.Layer(name="metasurface", thickness=0.2, epsilon=epsilon)
lay.plot()
plt.show()


##############################################################################
# Another lattice with non orthogonal basis vectors:


lat = nn.Lattice(basis_vectors=[[1, 0], [0.5, 0.5]], discretization=2**9)
epsilon = lat.ones()
scat = lat.ellipse(center=(0.75, 0.25), radii=(0.4, 0.1), rotate=15)
epsilon[scat] = 3 - 1j

lay = lat.Layer(name="grating", thickness=1.3, epsilon=epsilon)
ims = lay.plot(nper=(3, 2), show_cell=True, cmap="Greens", comp="im", cellstyle="y--")
plt.axis("off")
plt.colorbar(ims[0], orientation="horizontal")
plt.title(r"${\rm Im}\,\varepsilon$")
plt.tight_layout()
plt.show()
