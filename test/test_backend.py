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

    nn.set_backend("autograd")
    assert nn.numpy.__name__ == "autograd.numpy"
    assert nn.backend.__name__ == "autograd.numpy"
    assert nn.get_backend() == "autograd"

    nn.set_backend("jax")
    assert nn.numpy.__name__ == "jax.numpy"
    assert nn.backend.__name__ == "jax.numpy"
    assert nn.get_backend() == "jax"

    nn.set_backend("torch")
    assert nn.numpy.__name__ == "numpy"
    if nn.has_torch():
        assert nn.get_backend() == "torch"
        assert nn.backend.__name__ == "torch"

    with pytest.raises(ValueError) as excinfo:
        nn.set_backend("fake")
    assert "Unknown backend" in str(excinfo.value)
    nn.set_backend("numpy")
    assert nn.numpy.__name__ == "numpy"
    assert nn.backend.__name__ == "numpy"
    assert nn.get_backend() == "numpy"
