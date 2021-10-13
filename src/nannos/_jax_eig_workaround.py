#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

from jax.config import config

config.update("jax_enable_x64", True)
from jax import numpy as npj


def null_jax(A, eps=1e-11):
    u, s, vh = npj.linalg.svd(A, full_matrices=False)
    null_mask = s <= eps
    null_space = npj.compress(null_mask, vh, axis=0)
    return npj.transpose(null_space)


def eig_jax(A, eps=1e-11):
    l = npj.linalg.eigvals(A)
    N = A.shape[0]
    v = []
    for j in range(N):
        aj = null_jax(A - l[j] * npj.eye(N), eps=eps)
        v.append(aj)
    return l, npj.conj(npj.array(v)[:, :, 0].T)
