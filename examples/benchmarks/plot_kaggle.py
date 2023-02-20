#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


"""
Benchmarks
==========

Backend performace comparison.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

##############################################################################
# Load data
absolute_path = os.path.dirname("__file__")
full_path = os.path.join(absolute_path, "kaggle/nannos/results_kaggle.npz")

arch = np.load(full_path, allow_pickle=True)
num_harmo = arch["num_harmo_real"]
timings_gpu_torch = arch["timings_gpu_torch"]
timings_cpu_torch = arch["timings_cpu_torch"]
timings_numpy = arch["timings_numpy"]

##############################################################################
# Time vs. number of harmonics
# ------------------------------

tav_gpu_torch = np.array([h.average for h in timings_gpu_torch])
tstd_gpu_torch = np.array([h.stdev for h in timings_gpu_torch])
t_gpu_torch = np.array([h.all_runs for h in timings_gpu_torch])
tav_cpu_torch = np.array([h.average for h in timings_cpu_torch])
tstd_cpu_torch = np.array([h.stdev for h in timings_cpu_torch])
t_cpu_torch = np.array([h.all_runs for h in timings_cpu_torch])
tav_numpy = np.array([h.average for h in timings_numpy])
tstd_numpy = np.array([h.stdev for h in timings_numpy])
t_numpy = np.array([h.all_runs for h in timings_numpy])

plt.figure()
plt.errorbar(num_harmo, tav_gpu_torch, yerr=tstd_gpu_torch, label="torch GPU")
plt.errorbar(num_harmo, tav_cpu_torch, yerr=tstd_cpu_torch, label="torch CPU")
plt.errorbar(num_harmo, tav_numpy, yerr=tstd_numpy, label="numpy")
plt.xlabel("Number of harmonics")
plt.ylabel("CPU time (s)")
plt.yscale("log")
plt.legend()
plt.tight_layout()

##############################################################################
# Speedup vs. number of harmonics
# -----------------------------------

speedup_gpu = t_numpy / t_gpu_torch
speedup_gpu_av = np.mean(speedup_gpu, axis=1)
speedup_gpu_std = np.std(speedup_gpu, axis=1)
speedup_cpu = t_numpy / t_cpu_torch
speedup_cpu_av = np.mean(speedup_cpu, axis=1)
speedup_cpu_std = np.std(speedup_cpu, axis=1)


plt.figure()
plt.errorbar(num_harmo, speedup_gpu_av, yerr=speedup_gpu_std, label="torch GPU")
plt.errorbar(num_harmo, speedup_cpu_av, yerr=speedup_cpu_std, label="torch CPU")
plt.xlabel("number of harmonics")
plt.ylabel("Speedup vs. numpy")
plt.legend()
plt.tight_layout()
