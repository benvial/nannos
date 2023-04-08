#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


"""
Benchmark 2
===========

Backend performace comparison, with GPU acceleration, as run on Kaggle (https://www.kaggle.com/code/benjaminvial/nannos_cpu)
"""


import os

import matplotlib.pyplot as plt
import numpy as np

##############################################################################
# Load data
absolute_path = os.path.dirname("__file__")
full_path = os.path.join(absolute_path, "kaggle/nannos_cpu/results_kaggle_cpu.npz")
arch = np.load(full_path, allow_pickle=True)
num_harmo = arch["num_harmo_real"]
timings = arch["timedict"].item()
# cases = arch["cases"].tolist()


full_path = os.path.join(absolute_path, "kaggle/nannos_gpu/results_kaggle_gpu.npz")
arch = np.load(full_path, allow_pickle=True)
timings.update(arch["timedict"].item())
# cases += arch["cases"].tolist()

skip_first = True

ifirst = 1 if skip_first else 0
cases = ["numpy", "scipy", "autograd", "torch cpu", "torch gpu", "jax cpu", "jax gpu"]


##############################################################################
# Time vs. number of harmonics
# ------------------------------
tav = {}
tstd = {}
for case in cases:
    t = np.array(timings[case])[:, ifirst:]
    tav[case] = np.mean(t, axis=1)
    tstd[case] = np.std(t, axis=1)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

colors = colors[:3] + [colors[4], colors[4]] + [colors[5], colors[5]]

plt.figure()
for i, case in enumerate(cases):
    ls = "--" if case.split(" ")[-1] == "gpu" else "-"
    plt.errorbar(
        num_harmo,
        tav[case],
        yerr=tstd[case],
        label=case,
        ls=ls,
        color=colors[i],
        fmt="s",
        capsize=5,
    )

plt.xlabel("Number of harmonics")
plt.ylabel("CPU time (s)")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.tight_layout()

##############################################################################
# Speedup vs. number of harmonics
# -----------------------------------

speedup_av = {}
speedup_std = {}

for case in cases[1:]:
    s = np.array(timings["numpy"])[:, ifirst:] / np.array(timings[case])[:, ifirst:]
    speedup_av[case] = np.mean(s, axis=1)
    speedup_std[case] = np.std(s, axis=1)


plt.figure()
for i, case in enumerate(cases[1:]):
    ls = "--" if case.split(" ")[-1] == "gpu" else "-"
    plt.errorbar(
        num_harmo,
        speedup_av[case],
        yerr=speedup_std[case],
        label=case,
        ls=ls,
        color=colors[i + 1],
        fmt="s",
        capsize=5,
    )
plt.xlabel("number of harmonics")
plt.ylabel("Speedup vs. numpy")
plt.legend()
plt.tight_layout()
