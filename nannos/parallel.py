#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


__all__ = ["parloop"]

from functools import wraps

from joblib import Parallel, delayed


def parloop(n_jobs=1, prefer=None):
    def deco_parloop(func):
        """
        Decorate a function to parallelize.
        """

        @wraps(func)
        def my_func(*args, **kwargs):
            other_args = args[1:]
            return Parallel(n_jobs=n_jobs, prefer=prefer)(
                delayed(func)(x, *other_args, **kwargs) for x in args[0]
            )

        return my_func

    return deco_parloop
