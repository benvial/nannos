#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


from .. import backend as bk
from ..constants import pi
from ..utils import norm


def get_jones_field(t):
    norm_t = norm(t)
    n = [-t[1], t[0]]
    theta_J = bk.arccos(t[0] / norm_t)
    phi_J = pi / 8 * (1 + bk.cos(pi * norm_t))
    J = [
        bk.exp(1j * theta_J)
        / norm_t
        * (t[i] * bk.cos(phi_J) + 1j * n[i] * bk.sin(phi_J))
        for i in range(2)
    ]
    return J
