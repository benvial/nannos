#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: GPLv3


from IPython import get_ipython

from .jupyter import VersionTable

_IP = get_ipython()
if _IP is not None:
    _IP.register_magics(VersionTable)
