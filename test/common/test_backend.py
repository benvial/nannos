#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

import pytest


def test_backend():
    import nannos as nn

    assert nn.get_backend() == "numpy"
    assert nn.BACKEND == "numpy"

    nn.set_backend("scipy")
    assert nn.numpy.__name__ == "numpy"
    assert nn.backend.__name__ == "numpy"
    assert nn.get_backend() == "scipy"
    assert nn.BACKEND == "scipy"

    nn.set_backend("autograd")
    assert nn.numpy.__name__ == "autograd.numpy"
    assert nn.backend.__name__ == "autograd.numpy"
    assert nn.get_backend() == "autograd"
    assert nn.BACKEND == "autograd"

    nn.set_backend("jax")
    assert nn.numpy.__name__ == "jax.numpy"
    assert nn.backend.__name__ == "jax.numpy"
    assert nn.get_backend() == "jax"
    assert nn.BACKEND == "jax"

    nn.set_backend("torch")
    assert nn.numpy.__name__ == "numpy"
    if nn.has_torch():
        assert nn.get_backend() == "torch"
        assert nn.backend.__name__ == "torch"
        assert nn.BACKEND == "torch"

    with pytest.raises(ValueError) as excinfo:
        nn.set_backend("fake")
    assert "Unknown backend" in str(excinfo.value)
    nn.set_backend("numpy")
    assert nn.numpy.__name__ == "numpy"
    assert nn.backend.__name__ == "numpy"
    assert nn.get_backend() == "numpy"
    assert nn.BACKEND == "numpy"


def test_notorch(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    import nannos

    nannos.set_backend("torch")

    nannos.use_gpu(True)
    nannos.use_gpu(False)


def test_gpu(monkeypatch):
    import nannos

    nannos.set_backend("torch")
    nannos.use_gpu(True)
    nannos.use_gpu(False)
