# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.com
#  Copyright (C) 2015-2025 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
import time
from typing import Callable


class FixedIntervalProgressTracker(object):
    def __init__(self, interval: float, progress_callback: Callable):
        self.interval = interval
        self.last_update = 0
        self.progress_callback = progress_callback

    def update(self):
        now = time.time()
        if now - self.last_update >= self.interval:
            self.last_update = now
            self.progress_callback()

    def __call__(self):
        self.update()
