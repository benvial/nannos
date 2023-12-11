#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


__all__ = ["tic", "toc"]

import time


def tic():
    return time.time()


def toc(t0, verbose=True):
    t = time.time() - t0
    if verbose:
        print(f"elapsed time {t}s")
    return t
