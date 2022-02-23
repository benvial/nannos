#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of refidx
# License: GPLv3
# See the documentation at benvial.gitlab.io/refidx


def test_metadata(monkeypatch):
    import nannos

    vt = nannos.utils.VersionTable()
    vt.nannos_version_table()
