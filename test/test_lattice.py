#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


import numpy as np
import pytest

from nannos.lattice import *
from nannos.lattice import Lattice


def test_lattice():
    l = Lattice(((1, 2), (3, 4)))
    assert np.allclose(np.array([[1, 3], [2, 4]]), l.matrix)
    assert np.allclose(
        np.array([[-12.56637061, 6.28318531], [9.42477796, -3.14159265]]), l.reciprocal
    )


def test_truncate():
    l = Lattice(((1, 2), (3, 4)))
    g, nh = l.get_harmonics(100)
    g, nh = l.get_harmonics(100, method="parallelogrammic")
    with pytest.raises(ValueError) as excinfo:
        l.get_harmonics(100, method="unknown")
    assert "Unknown truncation method" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        l.get_harmonics(1.2)
    assert "nh must be integer." == str(excinfo.value)
