# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2022 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
import time
from enum import Enum
from typing import Callable, Optional, Tuple


class ConnectionStatus(Enum):
    connection_ok = 1
    connection_failed = 2
    connection_restored = 3


class ServerConnectionMonitor(object):
    """
    This class provides methods to monitor server connection status using particular failure detection scheme.
    """

    def __init__(self, ping_interval: float):
        self.ping_interval = ping_interval
        self._has_server_connection = True
        self._last_beat = float("-inf")
        self.disconnect_time = 0
        self.disconnect_reason = None

    @property
    def has_server_connection(self) -> bool:
        return self._has_server_connection

    def reset(self):
        self.disconnect_time = 0
        self.disconnect_reason = None

    def connection_failed(self, failure_reason: str):
        if self._has_server_connection:
            # save the first disconnection time and reason
            self.disconnect_time = time.time()
            self.disconnect_reason = failure_reason

        self._has_server_connection = False

    def tick(
        self, connection_probe: Callable[..., Tuple[bool, Optional[str]]]
    ) -> ConnectionStatus:
        """Invoked at each appropriate execution tick. If appropriate, this method will attempt to check
        connectivity using provided connection probe callable."""
        next_beat = self._last_beat + self.ping_interval
        now = time.time()
        if next_beat <= now:
            self._last_beat = now
            success, reason = connection_probe()
            return self._on_ping_result(success, reason)
        elif self.has_server_connection:
            return ConnectionStatus.connection_ok
        else:
            return ConnectionStatus.connection_failed

    def _on_ping_result(
        self, success: bool, failure_reason: Optional[str]
    ) -> ConnectionStatus:
        """Invoked to process ping result"""
        if success:
            if not self.has_server_connection:
                status = ConnectionStatus.connection_restored
            else:
                status = ConnectionStatus.connection_ok

            self._has_server_connection = True
        else:
            status = ConnectionStatus.connection_failed
            self.connection_failed(failure_reason)

        return status
