#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: GPLv3


import os

backends = ["numpy", "scipy", "autograd", "jax", "torch"]
inputs = [(b, "cpu") for b in backends]
inputs += [(b, "gpu") for b in ["jax", "torch"]]

for i in inputs:
    backend, device = i
    fname = f"benchmark_{backend}_{device}.npz"
    os.system(f"modal volume get nannos-volume {fname} --force")
