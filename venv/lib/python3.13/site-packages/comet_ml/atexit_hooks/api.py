# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2023 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************

import atexit
import logging
import sys
import threading
from typing import Callable

from . import conditional_callable

LOGGER = logging.getLogger(__name__)


def register(callable: Callable) -> conditional_callable.ConditionalCallable:
    conditional_callable_ = conditional_callable.ConditionalCallable(
        callable, thread_exclusive=True
    )

    if hasattr(threading, "_register_atexit"):
        try:
            LOGGER.debug("Using threading._register_atexit hooking mechanism")
            threading._register_atexit(conditional_callable_)

            return conditional_callable_
        except Exception:
            LOGGER.debug(
                "Failed to use threading._atexit_register. Fallback to atexit.register"
                "It could be caused by changes in threading module. Python version %s",
                sys.version,
                exc_info=True,
            )

    LOGGER.debug("Using atexit.register hooking mechanism")
    atexit.register(conditional_callable_)

    return conditional_callable_
