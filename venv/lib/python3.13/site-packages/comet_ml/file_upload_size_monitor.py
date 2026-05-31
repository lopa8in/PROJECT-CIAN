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
from typing import Any


class UploadSizeMonitor(object):
    __slots__ = ["total_size", "bytes_read"]

    def __init__(self):
        self.total_size = None
        self.bytes_read = 0

    def monitor_callback(self, monitor: Any) -> None:
        self.bytes_read = monitor.bytes_read

    def reset(self) -> None:
        self.bytes_read = 0
