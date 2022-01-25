#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


__all__ = ["adaptive_sampler"]

from . import numpy as np
from .parallel import parloop


def adaptive_sampler(
    f, x, max_bend=10, max_x_rel=1e-3, max_df=0.05, parallel=False, n_jobs=1
):

    if parallel:

        @parloop(n_jobs=n_jobs)
        def _function_adapted(x):
            return f(x)

    else:

        def _function_adapted(x):
            return [f(_x) for _x in x]

    cmax = np.cos(max_bend * np.pi / 180)

    def get_new(x, y):
        x = np.sort(x)
        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)

        new_x = []
        for i in range(len(x) - 2):
            x_tmp = x[i : i + 3]
            y_tmp = y[i : i + 3]
            xp, x0, xn = x_tmp
            yp, y0, yn = y_tmp

            min_dx = max_x_rel * (x_max - x_min)
            min_dy = max_df * (y_max - y_min)

            ref_x = xn - x0 < min_dx and x0 - xp < min_dx
            ref_y = abs(y0 - yp) < min_dy and abs(yn - y0) < min_dy

            local_y_min = np.min(y_tmp)
            local_y_max = np.max(y_tmp)

            dx0 = (x0 - xp) / (xn - xp)
            dx1 = (xn - x0) / (xn - xp)
            dy0 = (y0 - yp) / (local_y_max - local_y_min)
            dy1 = (yn - y0) / (local_y_max - local_y_min)
            bend = (dx0 * dx1 + dy0 * dy1) / np.sqrt(
                (dx0 * dx0 + dy0 * dy0) * (dx1 * dx1 + dy1 * dy1)
            )
            bending = (bend) < cmax or dx1 > 3 * dx0 or dx0 > 3 * dx1

            refine = bending and not ref_x and not ref_y
            if refine:
                seg = []
                if x0 - xp < min_dx:
                    isegment = 1
                    seg.append(isegment)
                if xn - x0 < min_dx:
                    isegment = 0
                    seg.append(isegment)
                if x0 - xp > xn - x0:
                    isegment = 0
                else:
                    isegment = 1
                seg.append(isegment)
                seg = np.unique(seg)
                for isegment in seg:
                    x_new = 0.5 * sum(x_tmp[isegment : isegment + 2])
                    new_x.append(x_new)
        return np.unique(new_x)

    y = _function_adapted(x)
    if hasattr(y[0], "__len__") and len(y[0]) > 0:
        y_monitor = [_[0] for _ in y]
        multi_output = True
    else:
        multi_output = False
        y_monitor = y.copy()
    while True:
        old_x = x.tolist()
        if multi_output:
            new_x = get_new(x, y_monitor)
        else:
            new_x = get_new(x, y)
        if len(new_x) == 0:
            break

        x = np.hstack([x, new_x])
        x, iu = np.unique(x, return_index=True)
        q = [_x for _x in x if _x not in old_x]

        new_y = _function_adapted(q)
        if multi_output:
            y = np.vstack([y, new_y])
        else:
            y = np.hstack([y, new_y])

        y = y[iu]

        if multi_output:
            new_y_monitor = [_[0] for _ in new_y]
            y_monitor = np.hstack([y_monitor, new_y_monitor])
            y_monitor = y_monitor[iu]

    return x, y
