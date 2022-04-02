#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


from .. import backend as bk
from .. import get_backend
from ..constants import pi
from ..utils import norm

_arctan2 = bk.atan2 if get_backend() == "torch" else bk.arctan2


def get_jones_field(t):
    norm_t = norm(t)
    n = [-t[1], t[0]]
    theta = _arctan2(t[1], t[0])
    phi = pi / 8 * (1 + bk.cos(pi * norm_t))
    expo = bk.exp(1j * theta)
    J = [
        expo / norm_t * (t[i] * bk.cos(phi) + 1j * n[i] * bk.sin(phi)) for i in range(2)
    ]
    return J
