#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


"""
Geometry tools
==============

Defining patterns using shapely.
"""


import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as sg

from nannos.geometry import shape_mask

N = 2**9
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
epsilon = np.ones((N, N))

####################################################################
# A split ring resonator

t = 0.1
l_out = 0.8
l_arm = 0.45
gap = 0.05

s_out = (1 - l_out) / 2
s_in = s_out + t
l_in = l_out - 2 * t

sq_out = sg.Polygon(
    [
        (s_out, s_out),
        (s_out + l_out, s_out),
        (s_out + l_out, s_out + l_out),
        (s_out, s_out + l_out),
    ]
)

sq_in = sg.Polygon(
    [
        (s_in, s_in),
        (s_in + l_in, s_in),
        (s_in + l_in, s_in + l_in),
        (s_in, s_in + l_in),
    ]
)
ring = sq_out.difference(sq_in)

g = sg.Polygon(
    [
        (0.5 - gap / 2, s_out),
        (0.5 + gap / 2, s_out),
        (0.5 + gap / 2, s_out + t),
        (0.5 - gap / 2, s_out + t),
    ]
)
srr = ring.difference(g)


a = 0.5 - gap / 2
b = s_out + t
arm_left = sg.Polygon(
    [
        (a - t, b),
        (a - t, b + l_arm),
        (a, b + l_arm),
        (a, b),
    ]
)
a = 0.5 + gap / 2
arm_right = sg.Polygon(
    [
        (a, b),
        (a, b + l_arm),
        (a + t, b + l_arm),
        (a + t, b),
    ]
)

srr = srr.union(arm_left).union(arm_right)
mask = shape_mask(srr, x, y)
epsilon[mask] = 6


plt.imshow(epsilon, cmap="Pastel1_r", origin="lower", extent=[0, 1, 0, 1])
plt.colorbar()
plt.show()


####################################################################
# Various patterns


epsilon = np.ones((N, N))
circle = sg.Point(0.4, 0.4).buffer(0.3)
mask = shape_mask(circle, x, y)

epsilon[mask] = 2
circle1 = sg.Point(0.8, 0.8).buffer(0.1)
mask1 = shape_mask(circle1, x, y)
epsilon[mask1] = 3


circle3 = sg.Point(0.3, 0.3).buffer(0.2)
diff = circle.difference(circle3)
mask2 = shape_mask(diff, x, y)
epsilon[mask2] = 4


circle4 = sg.Point(0.3, 0.7).buffer(0.15)
test1 = diff.union(circle4)
mask3 = shape_mask(test1, x, y)
epsilon[mask3] = 5

polygon = sg.Polygon([(0.7, 0.1), (0.9, 0.1), (0.9, 0.4)])
mask4 = shape_mask(polygon, x, y)
epsilon[mask4] = 6

centers = [(0.1, 0.8), (0.5, 0.9), (0.8, 0.5)]
radii = [0.03, 0.08, 0.04]

for i, (c, r) in enumerate(zip(centers, radii)):
    circle = sg.Point(c).buffer(r)
    mask = shape_mask(circle, x, y)
    epsilon[mask] = 7 + i


plt.imshow(epsilon, cmap="Pastel1_r", origin="lower", extent=[0, 1, 0, 1])
plt.colorbar()
plt.show()
