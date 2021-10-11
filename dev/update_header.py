#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: GPLv3


import os

header = """#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io
"""


def rep_header(python_file, header):

    with open(python_file, "r") as f:
        lines = f.readlines()

    i = 0
    for line in lines:
        if line.startswith("#"):
            i += 1
        else:
            break

    header = header.splitlines()
    data = "\n".join(header) + "\n" + "".join(lines[i:])

    with open(python_file, "w") as f:
        f.write(data)


#
# python_file = "../src/nannos/log.py"
#
# rep_header(python_file)


def update(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_file = os.path.join(root, file)
                print(python_file)
                rep_header(python_file, header)


for directory in ["../src/nannos", "../tutorials", "../examples"]:
    update(directory)
