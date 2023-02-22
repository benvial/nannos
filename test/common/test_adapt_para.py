#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

import numpy as np

from nannos import adaptive_sampler

x = np.linspace(1, 3, 40)


def bumps(x):
    return (
        1 / np.abs((x - (1.3 - 0.1j))) ** 2 + 0.005 / np.abs((x - (2.4 - 0.01j))) ** 2
    )


def f(x):
    return bumps(x), bumps(x) * np.sin(2 * np.pi * x)


def test_all():
    all_x, all_y = adaptive_sampler(bumps, x)
    all_x_para, all_y_para = adaptive_sampler(bumps, x, n_jobs=2)
    assert np.allclose(all_x, all_x_para)
    assert np.allclose(all_y, all_y_para)

    all_x, all_y = adaptive_sampler(f, x)
    all_x_para, all_y_para = adaptive_sampler(f, x, n_jobs=2)
    assert np.allclose(all_x, all_x_para)
    assert np.allclose(all_y, all_y_para)
