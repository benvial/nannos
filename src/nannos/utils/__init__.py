#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

from IPython import get_ipython

from .helpers import *
from .jupyter import VersionTable
from .time import *

_IP = get_ipython()
if _IP is not None:
    _IP.register_magics(VersionTable)
