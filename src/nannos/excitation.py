#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

from . import backend as bk

__all__ = ["PlaneWave"]


class PlaneWave:
    """A plane wave object

    Parameters
    ----------
    frequency : float
        Frequency (the default is 1).
    angles : tuple
        Incidence angles :math:`(\\theta,\phi,\psi)` (the default is (0, 0, 0)).
        :math:`\\theta`: polar angle,
        :math:`\phi`: azimuthal angle,
        :math:`\psi`: polarization angle.


    """

    def __init__(self, frequency=1, angles=(0, 0, 0)):
        self.frequency = frequency
        self.angles = bk.array(angles, dtype=bk.float64)
        self.theta = bk.array(angles[0], dtype=bk.float64)
        self.phi = bk.array(angles[1], dtype=bk.float64)
        self.psi = bk.array(angles[2], dtype=bk.float64)

        k0 = 2 * bk.pi * frequency
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

        cx = -spsi * bk.cos(self.theta) * bk.cos(self.phi) - cpsi * bk.sin(self.phi)
        cy = -spsi * bk.cos(self.theta) * bk.sin(self.phi) + cpsi * bk.cos(self.phi)
        cz = -cpsi * bk.sin(self.theta)

        #
        # cx = cpsi * bk.cos(self.theta) * bk.cos(self.phi) - spsi * bk.sin(self.phi)
        # cy = cpsi * bk.cos(self.theta) * bk.sin(self.phi) + spsi * bk.cos(self.phi)
        # cz = -cpsi * bk.sin(self.theta)

        self.amplitude = bk.array([cx, cy, cz])


class CircPolPlaneWave(PlaneWave):
    def __init__(self, frequency=1, angles=(0, 0, 0), orientation="right"):
        super().__init__(frequency, angles)
        angles_rotated = bk.copy(angles).astype(float)
        angles_rotated[-1] += bk.pi / 2

        H = PlaneWave(frequency, angles)
        V = PlaneWave(frequency, angles_rotated)

        sign = +1 if orientation == "left" else -1

        self.amplitude = (H.amplitude + sign * 1j * V.amplitude) / 2**0.5


#
# frequency=1
# angles=(0, 0, 0)
# angles_rotated = bk.copy(angles).astype(float)
# angles_rotated[-1] += bk.pi/2
#
# H = PlaneWave(frequency, angles)
# V = PlaneWave(frequency, angles_rotated)
# print(H.amplitude)
# print(V.amplitude)
#
#
# R = CircPolPlaneWave(orientation="right")
# L = CircPolPlaneWave(orientation="left")
#
#
#
# print(R.amplitude)
# print(L.amplitude)
