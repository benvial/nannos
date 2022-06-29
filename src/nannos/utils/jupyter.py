#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

all = ["local_hardware_info", "VersionTable"]

import distutils.core
import os
import platform
import sys
import time

import pkg_resources
import psutil
from IPython.core.magic import Magics, line_magic, magics_class
from IPython.display import HTML, display

import nannos

dir_path = os.path.dirname(os.path.realpath(__file__))


def local_hardware_info():
    """Basic hardware information about the local machine.
    Gives actual number of CPU's in the machine, even when hyperthreading is
    turned on. CPU count defaults to 1 when true count can't be determined.
    Returns:
        dict: The hardware information.
    """
    results = {
        "python_compiler": platform.python_compiler(),
        "python_build": ", ".join(platform.python_build()),
        "python_version": platform.python_version(),
        "os": platform.system(),
        "memory": psutil.virtual_memory().total / (1024**3),
        "cpus": psutil.cpu_count(logical=False) or 1,
    }
    return results


@magics_class
class VersionTable(Magics):
    """A class of status magic functions."""

    @line_magic
    def nannos_version_table(self, line="", cell=None):
        """
        Print an HTML-formatted table with version numbers for Nannos and its
        dependencies. This should make it possible to reproduce the environment
        and the calculation later on.
        """

        html = "<h3>Version Information</h3>"
        html += "<table>"
        html += "<tr><th>Package</th></tr>"

        packages = []

        packages_names = ["nannos", "numpy", "scipy", "matplotlib"]

        for pkg in packages_names:
            ver = pkg_resources.get_distribution(pkg).version

            packages.append((f"<code>{pkg}</code>", ver))

        ver = pkg_resources.get_distribution("autograd").version
        packages.append(("<code>autograd</code>", ver))

        for name, version in packages:
            html += f"<tr><td>{name}</td><td>{version}</td></tr>"

        html += "<tr><th>System information</th></tr>"

        local_hw_info = local_hardware_info()
        sys_info = [
            ("Python version", local_hw_info["python_version"]),
            ("Python compiler", local_hw_info["python_compiler"]),
            ("Python build", local_hw_info["python_build"]),
            ("OS", "%s" % local_hw_info["os"]),
            ("CPUs", "%s" % local_hw_info["cpus"]),
            ("Memory (Gb)", "%s" % local_hw_info["memory"]),
        ]

        for name, version in sys_info:
            html += f"<tr><td>{name}</td><td>{version}</td></tr>"

        html += "<tr><td colspan='2'>%s</td></tr>" % time.strftime(
            "%a %b %d %H:%M:%S %Y %Z"
        )
        html += "</table>"

        return display(HTML(html))
