#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

"""
Benchmarks
==========

Backend performace comparison.
"""

import matplotlib.pyplot as plt
import numpy as np

plt.ion()
plt.close("all")

colors = ["#3b9dd4", "#ecd142", "#e87c40", "#b33dd1", "#50ba61", "#cd2323"]
markers = ["o", "s", "d", "v", "^", ">"]
figsize = (2, 2)
threads = [1, 2, 4, 8, 16]
devices = ["cpu", "gpu"]
backends = ["numpy", "scipy", "autograd", "jax", "torch"]

##############################################################################
# Time vs. number of harmonics
# ------------------------------

bench_threads = dict()

for num_threads in threads:
    plt.figure(figsize=figsize)
    bench = dict()
    i = 0
    for backend in backends:
        devdict = dict()
        for device in devices:
            g = "cuda" if device == "gpu" else device
            if not (
                device == "gpu" and backend in ["numpy", "scipy", "autograd", "jax"]
            ):
                arch = np.load(
                    f"{num_threads}/benchmark_{backend}_{g}.npz", allow_pickle=True
                )
                NH = arch["real_nh"]
                plt.plot(
                    arch["real_nh"],
                    arch["times"],
                    f"-{markers[i]}",
                    c=colors[i],
                    label=f"{backend} {device}",
                )
                devdict[device] = arch
                i += 1
            bench[backend] = devdict
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("number of harmonics")
    plt.ylabel("time (s)")
    plt.title(f"backends comparison {num_threads} threads")
    plt.tight_layout()

    bench_threads[num_threads] = bench


##############################################################################
# Speedup vs. number of harmonics
# ------------------------------

for num_threads in threads:
    plt.figure(figsize=figsize)
    arch_np = bench["numpy"]["cpu"]

    i = 1
    for device in devices:
        devdict_speedup = dict()
        for backend in backends:
            if not (
                device == "gpu" and backend in ["numpy", "scipy", "autograd", "jax"]
            ):
                arch = bench[backend][device]
                if backend != "numpy":
                    speedup = np.array(arch_np["times"]) / np.array(arch["times"])
                    plt.plot(
                        arch["real_nh"],
                        speedup,
                        f"-{markers[i]}",
                        c=colors[i],
                        label=f"{backend} {device}",
                    )
                    i += 1
    plt.legend()
    # plt.yscale("log")
    # plt.xscale("log")
    plt.xlabel("number of harmonics")
    plt.ylabel("speedup vs. numpy")
    plt.title(f"backends comparison {num_threads} threads")
    plt.tight_layout()


##############################################################################
# Time vs. number of threads
# -----------------------------


for inh in range(len(NH)):
    plt.figure(figsize=figsize)
    i = 0
    for device in devices:
        for backend in backends:
            t_threads = []
            for num_threads in threads:
                if not (
                    device == "gpu" and backend in ["numpy", "scipy", "autograd", "jax"]
                ):
                    arch = bench_threads[num_threads][backend][device]
                    t = arch["times"]
                    # t = np.array(t)
                    t_threads.append(t)
            if t_threads != []:
                t_threads = np.array(t_threads)
                plt.plot(
                    threads,
                    t_threads[:, inh],
                    f"-{markers[i]}",
                    c=colors[i],
                    label=f"{backend} {device}",
                )
                i += 1
    plt.xticks(threads)

    plt.legend(ncol=2)
    # plt.yscale("log")
    # plt.xscale("log")
    plt.xlabel("number of threads")
    plt.ylabel("time (s)")
    plt.title(f"backends comparison {NH[inh]} harmonics")
    plt.tight_layout()


##############################################################################
# Speedup vs. number of threads
# -----------------------------


for inh in range(len(NH)):
    plt.figure(figsize=figsize)
    i = 1
    for device in devices:
        for backend in backends:
            speedup_threads = []
            for num_threads in threads:
                if not (
                    device == "gpu" and backend in ["numpy", "scipy", "autograd", "jax"]
                ):
                    arch = bench_threads[num_threads][backend][device]
                    arch_np = bench_threads[num_threads]["numpy"]["cpu"]
                    if backend != "numpy":
                        t = arch["times"]
                        speedup = np.array(arch_np["times"]) / np.array(arch["times"])
                        speedup_threads.append(speedup)
            if speedup_threads != []:
                speedup_threads = np.array(speedup_threads)
                if backend != "numpy":
                    plt.plot(
                        threads,
                        speedup_threads[:, inh],
                        f"-{markers[i]}",
                        c=colors[i],
                        label=f"{backend} {device}",
                    )
                    i += 1
    plt.xticks(threads)
    plt.ylim(0.25, 3.8)

    plt.legend(ncol=2)
    # plt.yscale("log")
    # plt.xscale("log")
    plt.xlabel("number of threads")
    plt.ylabel("speedup vs. numpy")
    plt.title(f"backends comparison {NH[inh]} harmonics")
    plt.tight_layout()
