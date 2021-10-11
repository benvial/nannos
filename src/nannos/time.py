#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


__all__ = ["tic", "toc"]

import time


def tic():
    return time.time()


def toc(t0, verbose=True):
    t = time.time() - t0
    if verbose:
        print(f"elapsed time {t}s")
    return t
