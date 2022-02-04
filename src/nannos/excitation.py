#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

from . import numpy as np

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
        self.angles = np.array(angles, dtype=float)
        self.theta, self.phi, self.psi = angles

        k0 = 2 * np.pi * frequency
        self.wavenumber = k0
        self.wavevector = k0 * np.array(
            [
                np.sin(self.theta) * np.cos(self.phi),
                np.sin(self.theta) * np.sin(self.phi),
                np.cos(self.theta),
            ]
        )

        cpsi = np.cos(self.psi)  # p
        spsi = np.sin(self.psi)  # s

        cx = -spsi * np.cos(self.theta) * np.cos(self.phi) - cpsi * np.sin(self.phi)
        cy = -spsi * np.cos(self.theta) * np.sin(self.phi) + cpsi * np.cos(self.phi)
        cz = -cpsi * np.sin(self.theta)

        #
        # cx = cpsi * np.cos(self.theta) * np.cos(self.phi) - spsi * np.sin(self.phi)
        # cy = cpsi * np.cos(self.theta) * np.sin(self.phi) + spsi * np.cos(self.phi)
        # cz = -cpsi * np.sin(self.theta)

        self.amplitude = np.array([cx, cy, cz])


class CircPolPlaneWave(PlaneWave):
    def __init__(self, frequency=1, angles=(0, 0, 0), orientation="right"):
        super().__init__(frequency, angles)
        angles_rotated = np.copy(angles).astype(float)
        angles_rotated[-1] += np.pi / 2

        H = PlaneWave(frequency, angles)
        V = PlaneWave(frequency, angles_rotated)

        sign = +1 if orientation == "left" else -1

        self.amplitude = (H.amplitude + sign * 1j * V.amplitude) / 2 ** 0.5


#
# frequency=1
# angles=(0, 0, 0)
# angles_rotated = np.copy(angles).astype(float)
# angles_rotated[-1] += np.pi/2
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
