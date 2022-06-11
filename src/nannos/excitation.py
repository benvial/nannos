#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

from . import backend as bk
from .constants import pi

__all__ = ["PlaneWave"]

_deg2rad = pi / 180


class PlaneWave:
    r"""A plane wave object

    Parameters
    ----------
    wavelength : float
        Wavelength (the default is 1).
    angles : tuple
        Incidence angles :math:`(\theta,\phi,\psi)` in degrees (the default is (0, 0, 0)).
        :math:`\\theta`: polar angle,
        :math:`\phi`: azimuthal angle,
        :math:`\psi`: polarization angle.

    """

    def __init__(self, wavelength=1, angles=(0, 0, 0)):
        self.wavelength = bk.array(wavelength)
        self.angles_deg = bk.array(angles, dtype=bk.float64)
        self.angles = self.angles_deg * _deg2rad
        self.theta = bk.array(self.angles[0], dtype=bk.float64)
        self.phi = bk.array(self.angles[1], dtype=bk.float64)
        self.psi = bk.array(self.angles[2], dtype=bk.float64)
        self.frequency_scaled = bk.array(1 / self.wavelength)

        k0 = 2 * pi * self.frequency_scaled

        self.wavenumber = k0
        self.wavevector = k0 * bk.array(
            [
                bk.sin(self.theta) * bk.cos(self.phi),
                bk.sin(self.theta) * bk.sin(self.phi),
                bk.cos(self.theta),
            ]
        )

        cpsi = bk.cos(self.psi)  # p
        spsi = bk.sin(self.psi)  # s

        # cx = cpsi * bk.cos(self.theta) * bk.cos(self.phi) - spsi * bk.sin(self.phi)
        # cy = cpsi * bk.cos(self.theta) * bk.sin(self.phi) + spsi * bk.cos(self.phi)
        # cz = cpsi * bk.sin(self.theta)
        cx = cpsi * bk.cos(self.theta) * bk.cos(self.phi) - spsi * bk.sin(self.phi)
        cy = cpsi * bk.cos(self.theta) * bk.sin(self.phi) + spsi * bk.cos(self.phi)
        cz = -cpsi * bk.sin(self.theta)

        self.amplitude = bk.array([cx, cy, cz])

        kt = self.wavevector
        omega = k0
        K = bk.array([[kt[1] ** 2, -kt[0] * kt[1]], [-kt[0] * kt[1], kt[0] ** 2]])
        Q = omega**2 * bk.array(bk.eye(2)) - K
        q = (omega**2 - kt[0] ** 2 - kt[1] ** 2) ** 0.5
        C = bk.linalg.inv(Q) * (omega * q)
        et = bk.array([-self.amplitude[1], self.amplitude[0]])
        self.a0 = C @ et
        # self.a0 = -self.amplitude[1], self.amplitude[0]
        # self.a0 = self.amplitude[0], self.amplitude[1]
