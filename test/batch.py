#!/usr/bin/env python


import numpy as npo

import nannos as nn

Nx = 2**11
Ny = 2**11
nbatch = 11
# M=npo.random.rand(nbatch,Nx,Ny)
# t = nn.tic()
# Nf = nn.formulations.fft.fourier_transform(M)
# # Nf = npo.fft.fft2(M,axes=(-2, -1))
# nn.toc(t)
#
#
# t = nn.tic()
# for i in range(nbatch):
#     Nfloop = nn.formulations.fft.fourier_transform(M[i])
#     # Nfloop = npo.fft.fft2(M[i],axes=(-2, -1))
#     # print(npo.sum(npo.abs(Nf[i] - Nfloop)**2))
#     # assert npo.allclose(Nf[i],Nfloop)
#
# nn.toc(t)

Nx = Ny = 400

M0 = npo.random.rand(nbatch, Nx, Nx) + 1j * npo.random.rand(nbatch, Nx, Nx)

for backend in ["numpy", "torch"]:

    nn.set_backend(backend)
    M = nn.backend.array(M0)

    t = nn.tic()
    w, v = nn.backend.linalg.eig(M)
    nn.toc(t)
    t = nn.tic()
    for i in range(nbatch):
        wloop, vloop = nn.backend.linalg.eig(M[i])
        # Nfloop = npo.fft.fft2(M[i],axes=(-2, -1))
        # print(npo.sum(npo.abs(Nf[i] - Nfloop)**2))
        assert npo.allclose(w[i], wloop)

    nn.toc(t)
