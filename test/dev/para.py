#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

import matplotlib.pyplot as plt
import numpy as np

from nannos import adaptive_sampler

plt.ion()
plt.clf()

x = np.linspace(1, 3, 40)

import time


def bumps(x):
    return (
        1 / np.abs((x - (1.3 - 0.1j))) ** 2 + 0.005 / np.abs((x - (2.4 - 0.01j))) ** 2
    )


def f(x):
    print(x)
    # time.sleep(0.1)
    return bumps(x), bumps(x) * np.sin(2 * np.pi * x)


all_x, all_y = adaptive_sampler(f, x, parallel=True, n_jobs=8)

plt.clf()
plt.ylim(-10, 110)
plt.plot(all_x, all_y, "-")
