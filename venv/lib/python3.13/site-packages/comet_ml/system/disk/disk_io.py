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

DiskIOResults = namedtuple("DiskIOResults", ["disk_read_bps", "disk_write_bps"])


class DiskIOUtilizationProbe:
    def __init__(self):
        self.last_tick = 0.0
        self.last_bytes_read = 0
        self.last_bytes_write = 0

    def sample(self) -> Optional[DiskIOResults]:
        if psutil is None:
            return None

        counters = psutil.disk_io_counters()
        now = time.time()
        if self.last_tick == 0:
            self._save_current_state(
                time_now=now,
                bytes_read=counters.read_bytes,
                bytes_write=counters.write_bytes,
            )
            return None

        elapsed = now - self.last_tick
        read_bytes_rate = (counters.read_bytes - self.last_bytes_read) / elapsed
        write_bytes_rate = (counters.write_bytes - self.last_bytes_write) / elapsed

        self._save_current_state(
            time_now=now,
            bytes_read=counters.read_bytes,
            bytes_write=counters.write_bytes,
        )
        return DiskIOResults(
            disk_read_bps=read_bytes_rate, disk_write_bps=write_bytes_rate
        )

    def _save_current_state(self, time_now: float, bytes_read: int, bytes_write: int):
        self.last_tick = time_now
        self.last_bytes_read = bytes_read
        self.last_bytes_write = bytes_write
