#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


def test_metadata(monkeypatch):
    import nannos

    vt = nannos.utils.VersionTable()
    vt.nannos_version_table()
