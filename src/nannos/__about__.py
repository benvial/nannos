#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

try:
    # Python 3.8
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata

try:
    data = metadata.metadata("nannos")
    __version__ = metadata.version("nannos")
    __author__ = data.get("author")
    __description__ = data.get("summary")
except Exception:
    __version__ = "unknown"
    __author__ = "unknown"
    __description__ = "unknown"
