#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

import numpy as np
import pytest

np.random.seed(84)
N = 2 ** 10
matrix_fft = np.random.rand(N, N) + 1j * np.random.rand(N, N)


def test_fft():
    i = 0
    vals, vects = [], []
    for n in [8, 10, 12]:
        N = 2 ** n

        print(f"matrix size = {N}")
        matrix_fft = np.random.rand(N, N) + 1j * np.random.rand(N, N)
        for backend in ["autograd", "jax", "torch", "numpy"]:
            import nannos as nn

            print("backend: ", backend)
            nn.set_backend(backend)
            t = nn.tic()
            utf = nn.formulations.fft.fourier_transform(matrix_fft)
            nn.toc(t)
            vals.append(utf)
            # if i > 0:
            #     assert np.allclose(utf, vals[0])
            # i += 1
