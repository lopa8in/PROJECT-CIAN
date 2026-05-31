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
import time
from collections import namedtuple
from typing import Optional

try:
    import psutil
except Exception:
    psutil = None

NetworkRatesResult = namedtuple(
    "NetworkRatesResult", ["bytes_sent_rate", "bytes_recv_rate"]
)


class NetworkRatesProbe(object):
    def __init__(self):
        self.last_tick = 0.0
        self.last_bytes_recv = 0
        self.last_bytes_sent = 0

    def current_rate(self) -> Optional[NetworkRatesResult]:
        if psutil is None:
            return None

        counters = psutil.net_io_counters()
        now = time.time()
        if self.last_tick == 0.0:
            self._save_current_state(
                time_now=now,
                bytes_sent=counters.bytes_sent,
                bytes_recv=counters.bytes_recv,
            )
            return None

        elapsed = now - self.last_tick
        bytes_sent_rate = (counters.bytes_sent - self.last_bytes_sent) / elapsed
        bytes_recv_rate = (counters.bytes_recv - self.last_bytes_recv) / elapsed

        self._save_current_state(
            time_now=now, bytes_sent=counters.bytes_sent, bytes_recv=counters.bytes_recv
        )

        return NetworkRatesResult(
            bytes_recv_rate=int(bytes_recv_rate), bytes_sent_rate=int(bytes_sent_rate)
        )

    def _save_current_state(self, time_now: float, bytes_sent: int, bytes_recv: int):
        self.last_tick = time_now
        self.last_bytes_recv = bytes_recv
        self.last_bytes_sent = bytes_sent
