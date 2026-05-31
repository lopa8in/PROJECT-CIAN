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
import collections
import threading
from typing import Callable, Tuple

from . import callbacks


class StatusHandler:
    def __init__(
        self,
    ):
        self._lock = threading.Lock()
        self._completed_events = collections.defaultdict(list)
        self._failed_events = collections.defaultdict(list)

    def start_processing(
        self, name: str
    ) -> Tuple[Callable[[], None], Callable[[], None]]:
        with self._lock:
            uploaded_event = threading.Event()
            failed_event = threading.Event()
            self._completed_events[name].append(uploaded_event)
            self._failed_events[name].append(failed_event)

            return uploaded_event.set, failed_event.set

    def observer(self, name: str) -> callbacks.StatusObserver:
        with self._lock:
            if name not in self._completed_events:
                raise KeyError(
                    "Processing status for {} can't be observed because start_processing method wasn't called".format(
                        name
                    )
                )
            observer_callback = callbacks.StatusObserver(
                self._completed_events[name],
                self._failed_events[name],
                self._lock,
            )

            return observer_callback
