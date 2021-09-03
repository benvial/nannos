#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

from . import numpy as np

__all__ = ["PlaneWave"]


class PlaneWave:
    """A plane wave object.

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
        self.angles = angles
        self.theta, self.phi, self.psi = angles

        k0 = 2 * np.pi * frequency
        self.wavenumber = k0
        self.wavevector = k0 * np.array(
            (
                np.sin(self.theta) * np.cos(self.phi),
                np.sin(self.theta) * np.sin(self.phi),
                np.cos(self.theta),
            )
        )
        cpsi = np.cos(self.psi)  # p
        spsi = np.sin(self.psi)  # s
        cx = -spsi * np.cos(self.theta) * np.cos(self.phi) - cpsi * np.sin(self.phi)
        cy = -spsi * np.cos(self.theta) * np.sin(self.phi) + cpsi * np.cos(self.phi)
        cz = -cpsi * np.sin(self.theta)
        self.amplitude = [cx, cy, cz]
