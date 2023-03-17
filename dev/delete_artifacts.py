#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

import gitlab

gl = gitlab.Gitlab.from_config("gitlab")
projects = gl.projects.list(search="nannos", owned=True)
project = projects[0]
jobs = project.jobs.list(all=True)
jj = 0
for job in jobs:
    if (
        job.attributes["pipeline"]["status"] == "success"
        and job.attributes["stage"] == "deploy"
        and job.attributes["name"] == "pages"
    ):
        jj += 1
        if jj > 2:
            try:
                art = job.attributes["artifacts_file"]
                job.delete_artifacts()
            except KeyError:
                pass
