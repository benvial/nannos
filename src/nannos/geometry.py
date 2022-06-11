#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


"""
Geometry helpers
"""

# __all__ = ["shape_mask"]

import shapely.affinity as sa
import shapely.geometry as sg

from . import backend as bk
from . import get_backend

# from https://gist.github.com/perrette/a78f99b76aed54b6babf3597e0b331f8


def _grid_bbox(x, y):
    dx = dy = 0
    return x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2


def _bbox_to_rect(bbox):
    left, right, bottom, top = bbox
    return sg.Polygon([(left, bottom), (right, bottom), (right, top), (left, top)])


def set_index_2d(mat, val, idx1=(None, None), idx2=(None, None)):
    idx = slice(*idx1), slice(*idx2)
    if get_backend() == "jax":
        mat = mat.at[idx].set(val)
    else:
        mat[idx] = val
    return mat


def shape_mask(shp, x, y, m=None):
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
    >>> mask = shape_mask(poly, x, y)
    """
    rect = _bbox_to_rect(_grid_bbox(x, y))

    if m is None:
        m = bk.zeros((len(y), len(x)), dtype=bool)

    if not shp.intersects(rect):
        m = set_index_2d(m, False)

    elif shp.contains(rect):
        m = set_index_2d(m, True)

    else:
        lx, ly = m.shape

        if lx == 1 and ly == 1:

            val = shp.contains(sg.Point(x[0], y[0]))
            m = set_index_2d(m, val)

        elif lx == 1:

            val = shape_mask(shp, x[: ly // 2], y, m[:, : ly // 2])
            m = set_index_2d(m, val, idx2=(None, ly // 2))
            val = shape_mask(shp, x[ly // 2 :], y, m[:, ly // 2 :])
            m = set_index_2d(m, val, idx2=(ly // 2, None))

        elif ly == 1:
            val = shape_mask(shp, x, y[: lx // 2], m[: lx // 2])
            m = set_index_2d(m, val, idx1=(None, lx // 2))
            val = shape_mask(shp, x, y[lx // 2 :], m[lx // 2 :])
            m = set_index_2d(m, val, idx1=(lx // 2, None))

        else:
            val = shape_mask(shp, x[: ly // 2], y[: lx // 2], m[: lx // 2, : ly // 2])

            m = set_index_2d(m, val, (None, lx // 2), (None, ly // 2))
            val = shape_mask(shp, x[ly // 2 :], y[: lx // 2], m[: lx // 2, ly // 2 :])
            m = set_index_2d(m, val, (None, lx // 2), (ly // 2, None))
            val = shape_mask(shp, x[: ly // 2], y[lx // 2 :], m[lx // 2 :, : ly // 2])
            m = set_index_2d(m, val, (lx // 2, None), (None, ly // 2))
            val = shape_mask(shp, x[ly // 2 :], y[lx // 2 :], m[lx // 2 :, ly // 2 :])
            m = set_index_2d(m, val, (lx // 2, None), (ly // 2, None))

    return m


def geometry_mask(geom, lattice, Nx, Ny):
    x0 = bk.linspace(0, 1.0, Nx)
    y0 = bk.linspace(0, 1.0, Ny)
    x_, y_ = bk.meshgrid(x0, y0, indexing="ij")
    grid = bk.stack([x_, y_])
    x, y = grid[0][:, 0], grid[1][0, :]
    invM = bk.linalg.inv(lattice.matrix)
    matrix = invM.ravel().tolist() + [0, 0]
    geom = sa.affine_transform(geom, matrix)
    mask = shape_mask(geom, x, y).T
    return mask


def polygon(vertices, lattice, Nx, Ny):
    polygon = sg.Polygon(vertices)
    return geometry_mask(polygon, lattice, Nx, Ny)


def circle(center, radius, lattice, Nx, Ny):
    circ = sg.Point(*center).buffer(radius)
    return geometry_mask(circ, lattice, Nx, Ny)


def ellipse(center, radii, lattice, Nx, Ny, rotate=0):
    radius_x, radius_y = radii
    cent = sg.Point(*center)
    circ = cent.buffer(radius_x)
    ell = sa.scale(circ, xfact=1.0, yfact=radius_y / radius_x, zfact=1.0, origin=cent)
    if rotate != 0:
        ell = sa.rotate(ell, rotate, origin=cent, use_radians=False)
    return geometry_mask(ell, lattice, Nx, Ny)


def rectangle(center, widths, lattice, Nx, Ny, rotate=0):
    vertices = [[center[0] - widths[0] / 2, center[1] - widths[1] / 2]]
    vertices.append([center[0] + widths[0] / 2, center[1] - widths[1] / 2])
    vertices.append([center[0] + widths[0] / 2, center[1] + widths[1] / 2])
    vertices.append([center[0] - widths[0] / 2, center[1] + widths[1] / 2])
    rect = sg.Polygon(vertices)
    cent = sg.Point(*center)
    if rotate != 0:
        rect = sa.rotate(rect, rotate, origin=cent, use_radians=False)
    return geometry_mask(rect, lattice, Nx, Ny)


def square(center, width, lattice, Nx, Ny, rotate=0):
    return rectangle(center, (width, width), lattice, Nx, Ny, rotate)
