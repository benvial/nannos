#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


def test_nometadata():
    import importlib

    import nannos

    importlib.reload(nannos.__about__)


def test_data():
    import nannos

    nannos.__about__.get_meta(None)


def test_info():
    import nannos

    nannos.print_info()
