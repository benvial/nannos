#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


__all__ = ["adaptive_sampler"]

from . import numpy as np


def adaptive_sampler(f, z0, max_bend=10, max_z_rel=1e-3, max_df=0.05):

    z0 = np.sort(z0)
    zmin = min(z0)
    zmax = max(z0)

    tall = [f(z) for z in z0]
    z = z0.tolist()
    # t = t0.tolist()
    cmax = np.cos(max_bend * np.pi / 180)
    samp = True
    isamp = 0
    if hasattr(tall[0], "__len__") and len(tall[0]) > 0:
        t = [T[0] for T in tall]
        multi_output = True
    else:
        multi_output = False
        t = tall.copy()

    while samp:
        tmin = np.min(t)
        tmax = np.max(t)
        b = []
        for iz in range(len(z) - 2):
            ztmp = z[iz : iz + 3]
            ttmp = t[iz : iz + 3]

            xp, x0, xn = ztmp
            yp, y0, yn = ttmp

            min_dz = max_z_rel * (zmax - zmin)
            min_dt = max_df * (tmax - tmin)

            refx = xn - x0 < min_dz and x0 - xp < min_dz
            refy = abs(y0 - yp) < min_dt and abs(yn - y0) < min_dt

            local_y_max = yp
            if y0 > local_y_max:
                local_y_max = y0
            if yn > local_y_max:
                local_y_max = yn
            local_y_min = yp
            if y0 < local_y_min:
                local_y_min = y0
            if yn < local_y_min:
                local_y_min = yn
            dx0 = (x0 - xp) / (xn - xp)
            dx1 = (xn - x0) / (xn - xp)
            dy0 = (y0 - yp) / (local_y_max - local_y_min)
            dy1 = (yn - y0) / (local_y_max - local_y_min)

            bend = (dx0 * dx1 + dy0 * dy1) / np.sqrt(
                (dx0 * dx0 + dy0 * dy0) * (dx1 * dx1 + dy1 * dy1)
            )

            bending = (bend) < cmax or dx1 > 3 * dx0 or dx0 > 3 * dx1

            b.append(bending)
            if bending and not refx and not refy:
                seg = []
                if x0 - xp < min_dz:
                    isegment = 1
                    seg.append(isegment)
                if xn - x0 < min_dz:
                    isegment = 0
                    seg.append(isegment)
                if x0 - xp > xn - x0:
                    isegment = 0
                else:
                    isegment = 1
                seg.append(isegment)
                seg = np.unique(seg)

                for isegment in seg:
                    # isegment = np.random.randint(2)
                    znew = 0.5 * sum(ztmp[isegment : isegment + 2])
                    if znew not in z:
                        z.append(znew)
                        tnew = f(znew)
                        if multi_output:
                            t.append(tnew[0])
                            tall.append(tnew)
                        else:
                            t.append(tnew)

        z1 = np.array(z)
        t1 = np.array(t)
        ind = np.argsort(z1)
        z = z1[ind].tolist()
        t = t1[ind].tolist()
        if multi_output:
            tall = np.array(tall)[ind].tolist()
        samp = np.any(b)

        isamp += 1
        if isamp > 100:
            break

    tout = tall if multi_output else t
    return np.array(z), np.array(tout)
