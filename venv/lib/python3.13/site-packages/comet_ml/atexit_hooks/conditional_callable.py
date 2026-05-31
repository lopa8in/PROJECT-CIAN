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

import logging
import os
import threading
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)


class ConditionalCallable:
    def __init__(self, callable: Callable, thread_exclusive: bool) -> None:
        self._callable = callable

        self._thread_exclusive = thread_exclusive
        self._thread_id = threading.get_ident()
        self._process_id = os.getpid()

        self._enabled = True

    def disable(self) -> None:
        self._enabled = False
        self._callable = None  # to free the link for garbage collector

    def __call__(self) -> Any:
        if not self._enabled:
            return None

        called_from_original_thread = (
            self._thread_id == threading.get_ident() and self._process_id == os.getpid()
        )
        if self._thread_exclusive and not called_from_original_thread:
            LOGGER.debug(
                "Conditional callable execution wasn't allowed for process %d, thread %d, function %s because it's not the original thread",
                os.getpid(),
                threading.get_ident(),
                self._callable,
            )
            return None

        LOGGER.debug(
            "Conditional callable execution allowed for process %d, thread %d, function %s",
            os.getpid(),
            threading.get_ident(),
            self._callable,
        )
        return self._callable()
