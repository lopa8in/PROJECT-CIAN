# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.com
#  Copyright (C) 2015-2021 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
import threading
from typing import List

Status = str


class StatusObserver:
    def __init__(
        self,
        completed_events: List[threading.Event],
        failed_events: List[threading.Event],
        lock: threading.Lock,
    ):
        self._completed_events = completed_events
        self._failed_events = failed_events
        self._lock = lock

    def __call__(self) -> Status:
        with self._lock:
            all_failed = all(
                failed_event.is_set() for failed_event in self._failed_events
            )
            if all_failed:
                return "FAILED"

            any_completed = any(
                completed_event.is_set() for completed_event in self._completed_events
            )
            if any_completed:
                return "COMPLETED"

            return "IN_PROGRESS"
