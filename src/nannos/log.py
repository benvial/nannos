#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


__all__ = ["set_log_level", "logger"]


import logging

from colorlog import ColoredFormatter

LEVELS = dict(
    DEBUG=logging.DEBUG,
    INFO=logging.INFO,
    WARNING=logging.WARNING,
    ERROR=logging.ERROR,
    CRITICAL=logging.CRITICAL,
)


def set_log_level(level):
    """Sets the log level.

    Parameters
    ----------
    level : str
        The verbosity level.
        Valid values are ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR`` or ``CRITICAL``.

    """

    global logger
    LOG_LEVEL = LEVELS[level]
    LOGFORMAT = (
        "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
    )

    logging.root.setLevel(LOG_LEVEL)
    formatter = ColoredFormatter(LOGFORMAT)
    stream = logging.StreamHandler()
    stream.setLevel(LOG_LEVEL)
    stream.setFormatter(formatter)
    logger = logging.getLogger("pythonConfig")
    logger.setLevel(LOG_LEVEL)
    [logger.removeHandler(h) for h in logger.handlers]
    logger.addHandler(stream)


set_log_level("WARNING")
