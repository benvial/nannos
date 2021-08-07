#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import pytest


def test_backend():
    import nannos

    nannos.set_backend("numpy")
    assert nannos.numpy.__name__ == "numpy"
    nannos.set_backend("autograd")
    assert nannos.numpy.__name__ == "autograd.numpy"
    with pytest.raises(ValueError) as excinfo:
        nannos.set_backend("fake")
    assert "Unknown backend" in str(excinfo.value)
