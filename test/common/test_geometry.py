#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


def test_geo():

    import shapely.geometry as sg

    import nannos as nn
    import nannos.geometry as ng

    bk = nn.backend

    N = 2 ** 7
    x = bk.linspace(0, 1, N)
    y = bk.linspace(0, 1, N)
    epsilon = bk.ones((N, N))
    circle = sg.Point(0.4, 0.4).buffer(0.3)
    mask = ng.shape_mask(circle, x, y)
    epsilon[mask] = 2

    poly = sg.Point(0, 0).buffer(1)
    x = bk.linspace(-5, 5, 100)
    y = bk.linspace(-5, 5, 100)
    mask = ng.outline_to_mask(poly.boundary, x, y)
