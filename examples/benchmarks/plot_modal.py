#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: GPLv3


"""
Benchmark Modal
===============

Numerical backends performace comparison, with GPU acceleration, as run on Modal (https://modal.com)
"""


import os

import matplotlib.pyplot as plt
import numpy as np

backends = ["numpy", "scipy", "autograd", "jax", "torch"]
inputs = [(b, "cpu") for b in backends]
inputs += [(b, "gpu") for b in ["jax", "torch"]]


t = []

plt.figure()

for i in inputs:
    backend, device = i
    fname = f"modal/benchmark_{backend}_{device}.npz"
    arch = np.load(fname)
    times = arch["times"]
    times_all = arch["times_all"]
    real_nh = arch["real_nh"]
    nh = arch["nh"]

    t.append(times_all)

    plt.plot(real_nh, times_all, "-o", label=f"{backend} {device}")
plt.legend()
plt.xlabel("number of harmonics")
plt.ylabel("time [s]")
plt.xscale("log")
plt.yscale("log")


plt.figure()
for j, i in enumerate(inputs):
    backend, device = i
    plt.plot(real_nh, t[0] / t[j], "-o", label=f"{backend} {device}")
plt.legend()
plt.xlabel("number of harmonics")
plt.ylabel("speedup vs numpy")
plt.xscale("log")
# plt.yscale("log")
