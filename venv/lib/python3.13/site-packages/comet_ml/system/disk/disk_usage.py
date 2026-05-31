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
from collections import namedtuple
from typing import Optional

try:
    import psutil
except Exception:
    psutil = None

DiskUsageResults = namedtuple(
    "DiskUsageResults", ["disk_used_bytes", "disk_used_percent"]
)


class DiskUsageProbe:
    def __init__(self, path: str = "/"):
        self.path = path
        self.disk_usage_bytes = 0
        self.disk_usage_percent = 0

    def sample(self) -> Optional[DiskUsageResults]:
        if psutil is None:
            return None

        self.disk_usage_bytes = psutil.disk_usage(self.path).used
        self.disk_usage_percent = psutil.disk_usage(self.path).percent
        return DiskUsageResults(
            disk_used_bytes=self.disk_usage_bytes,
            disk_used_percent=self.disk_usage_percent,
        )
