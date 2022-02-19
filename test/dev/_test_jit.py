#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

# import nannos as nn
# nn.set_backend("jax")
# from nannos import numpy as np
# from nannos import grad
# from jax import jit
#
# import random
# import pytest
# import numpy as npo
#
#
# random.seed(84)
# N = 300
# l = nn.Layer("test", 1)
#
# matrix = npo.random.rand(N, N) + 1j * npo.random.rand(N, N)
# w, v = l.solve_eigenproblem(matrix)
#
# def fnp(matrix):
#     eig_func = npo.linalg.eig
#     w, v = eig_func(matrix)
#     q = w ** 0.5
#     q = npo.where(np.imag(q) < 0.0, -q, q)
#     return q, v
#
#
# def f(matrix):
#     return l.solve_eigenproblem(matrix)
#
#
# fjit = jit(f)
#
#
# wjit, vjit = fjit(matrix)
