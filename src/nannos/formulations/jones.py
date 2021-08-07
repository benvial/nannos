#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


from .. import numpy as np
from ..constants import pi
from ..helpers import norm


def get_jones_field(t):
    norm_t = norm(t)
    n = np.array([-t[1], t[0]])
    theta_J = np.arccos(t[0] / norm_t)
    phi_J = pi / 8 * (1 + np.cos(pi * norm_t))
    J = np.exp(1j * theta_J) / norm_t * (t * np.cos(phi_J) + 1j * n * np.sin(phi_J))
    return J
