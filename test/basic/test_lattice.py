#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

import numpy as npo
import pytest

import nannos as nn
from nannos.utils import allclose


def test_lattice():
    lat = nn.Lattice(((1, 2), (3, 4)))
    assert allclose(npo.array([[1, 3], [2, 4]]), lat.matrix)
    assert allclose(
        npo.array([[-12.56637061, 6.28318531], [9.42477796, -3.14159265]]),
        lat.reciprocal,
    )


def test_truncate():
    lat = nn.Lattice(((1, 2), (3, 4)))
    g, nh = lat.get_harmonics(100)
    lat = nn.Lattice(((1, 2), (3, 4)), truncation="parallelogrammic")
    g, nh = lat.get_harmonics(100)

    with pytest.raises(ValueError) as excinfo:
        lat.get_harmonics(1.2)
    assert "nh must be integer." == str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        lat = nn.Lattice(((1, 2), (3, 4)), truncation="whatever")
    assert "Unknown truncation method" in str(excinfo.value)
