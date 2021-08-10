#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

__all__ = ["set_log_level", "log"]


import logging

from colorlog import ColoredFormatter

LEVELS = dict(
    DEBUG=logging.DEBUG,
    INFO=logging.INFO,
    WARNING=logging.WARNING,
    ERROR=logging.ERROR,
    CRITICAL=logging.CRITICAL,
)


def set_log_level(level="INFO"):
    """Sets the log level.

    Parameters
    ----------
    level : str
        The verbosity level.
        Valid values are ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR`` or ``CRITICAL``
        (the default is ``INFO``).

    """

    global log
    LOG_LEVEL = LEVELS[level]
    LOGFORMAT = (
        "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
    )

    logging.root.setLevel(LOG_LEVEL)
    formatter = ColoredFormatter(LOGFORMAT)
    stream = logging.StreamHandler()
    stream.setLevel(LOG_LEVEL)
    stream.setFormatter(formatter)
    log = logging.getLogger("pythonConfig")
    log.setLevel(LOG_LEVEL)
    # if not log.hasHandlers():
    [log.removeHandler(h) for h in log.handlers]
    log.addHandler(stream)


set_log_level("INFO")
