#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


from shapely.geometry import Polygon

from nannos import numpy as np

# from https://gist.github.com/perrette/a78f99b76aed54b6babf3597e0b331f8


def outline_to_mask(line, x, y):
    """Create mask from outline contour

    Parameters
    ----------
    line: array-like (N, 2)
    x, y: 1-D grid coordinates (input for meshgrid)

    Returns
    -------
    mask : 2-D boolean array (True inside)

    Examples
    --------
    >>> from shapely.geometry import Point
    >>> poly = Point(0,0).buffer(1)
    >>> x = np.linspace(-5,5,100)
    >>> y = np.linspace(-5,5,100)
    >>> mask = outline_to_mask(poly.boundary, x, y)
    """
    import matplotlib.path as mplp

    mpath = mplp.Path(line)
    X, Y = np.meshgrid(x, y)
    points = np.array((X.flatten(), Y.flatten())).T
    mask = mpath.contains_points(points).reshape(X.shape)
    return mask


def _grid_bbox(x, y):
    dx = dy = 0
    return x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2


def _bbox_to_rect(bbox):
    l, r, b, t = bbox
    return Polygon([(l, b), (r, b), (r, t), (l, t)])


def shp_mask(shp, x, y, m=None):
    """Use recursive sub-division of space and shapely contains method to create a raster mask on a regular grid.

    Parameters
    ----------
    shp : shapely's Polygon (or whatever with a "contains" method and intersects method)
    x, y : 1-D numpy arrays defining a regular grid
    m : mask to fill, optional (will be created otherwise)

    Returns
    -------
    m : boolean 2-D array, True inside shape.

    Examples
    --------
    >>> from shapely.geometry import Point
    >>> poly = Point(0,0).buffer(1)
    >>> x = np.linspace(-5,5,100)
    >>> y = np.linspace(-5,5,100)
    >>> mask = shp_mask(poly, x, y)
    """
    rect = _bbox_to_rect(_grid_bbox(x, y))

    if m is None:
        m = np.zeros((y.size, x.size), dtype=bool)

    if not shp.intersects(rect):
        m[:] = False

    elif shp.contains(rect):
        m[:] = True

    else:
        k, l = m.shape

        if k == 1 and l == 1:
            m[:] = shp.contains(Point(x[0], y[0]))

        elif k == 1:
            m[:, : l // 2] = shp_mask(shp, x[: l // 2], y, m[:, : l // 2])
            m[:, l // 2 :] = shp_mask(shp, x[l // 2 :], y, m[:, l // 2 :])

        elif l == 1:
            m[: k // 2] = shp_mask(shp, x, y[: k // 2], m[: k // 2])
            m[k // 2 :] = shp_mask(shp, x, y[k // 2 :], m[k // 2 :])

        else:
            m[: k // 2, : l // 2] = shp_mask(
                shp, x[: l // 2], y[: k // 2], m[: k // 2, : l // 2]
            )
            m[: k // 2, l // 2 :] = shp_mask(
                shp, x[l // 2 :], y[: k // 2], m[: k // 2, l // 2 :]
            )
            m[k // 2 :, : l // 2] = shp_mask(
                shp, x[: l // 2], y[k // 2 :], m[k // 2 :, : l // 2]
            )
            m[k // 2 :, l // 2 :] = shp_mask(
                shp, x[l // 2 :], y[k // 2 :], m[k // 2 :, l // 2 :]
            )

    return m


import shapely.geometry as sg

circle = sg.Point(0, 0).buffer(0.5)

N = 2 ** 9
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
mask = shp_mask(circle, x, y)

import matplotlib.pyplot as plt

plt.ion()

plt.clf()

# plt.imshow(mask)
# plt.colorbar()


epsilon = np.ones(mask.shape)

epsilon[(mask)] = 2


circle1 = sg.Point(0.5, 0.5).buffer(0.1)

mask1 = shp_mask(circle1, x, y)

epsilon[(mask1)] = 3


circle3 = sg.Point(0.1, -0.5).buffer(0.25)

test = circle.difference(circle3)

mask2 = shp_mask(test, x, y)

epsilon[(mask2)] = 4


circle4 = sg.Point(-0.2, 0.5).buffer(0.35)

test1 = circle.union(circle4)
mask3 = shp_mask(test1, x, y)
epsilon[(mask3)] = 5
plt.imshow(epsilon, cmap="Pastel1")
plt.colorbar()
