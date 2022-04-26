#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


import numpy as npo

import nannos as nn

npo.random.seed(1984)


def benchmark(f, N, nbatch):
    print("=================================")
    print(f"=============== {f} =============")
    print("=================================")
    M0 = npo.random.rand(nbatch, N, N) + 1j * npo.random.rand(nbatch, N, N)

    for backend in ["numpy", "torch"]:
        nn.set_backend(backend)
        for dev in ["cpu", "gpu"]:
            if dev == "gpu" and backend == "numpy":
                pass
            else:
                if dev == "gpu":
                    nn.use_gpu()

                if f == "fft":
                    F = nn.formulations.fft.fourier_transform
                elif f == "eig":
                    F = nn.backend.linalg.eig
                elif f == "inv":
                    F = nn.backend.linalg.inv
                else:
                    raise ValueError

                M = nn.backend.array(M0)

                if dev == "gpu":
                    I = F(M)

                print(">>> batch")
                t = nn.tic()
                I = F(M)
                tbatch = nn.toc(t)
                print(">>> loop")
                t = nn.tic()
                for i in range(nbatch):
                    F(M[i])
                tloop = nn.toc(t)

                print("--------------------------")

                print(f"speedup loop = {tloop/tbatch}")

                if backend != "numpy":
                    print(f"speedup numpy loop = {tloop_numpy/tloop}")
                    print(f"speedup numpy batch = {tbatch_numpy/tbatch}")
                else:
                    tloop_numpy = tloop
                    tbatch_numpy = tbatch

                if dev != "gpu":
                    tloop_torch = tloop
                    tbatch_torch = tbatch
                else:
                    print(f"speedup torch gpu loop = {tloop_torch/tloop}")
                    print(f"speedup torch gpu batch = {tbatch_torch/tbatch}")


benchmark("fft", 2**10, 10)
benchmark("eig", 300, 10)
benchmark("inv", 500, 50)
