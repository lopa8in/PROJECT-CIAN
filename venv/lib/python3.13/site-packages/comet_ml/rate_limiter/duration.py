# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2024 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
from enum import Enum


class TimeUnit(Enum):
    SECOND = 1
    MINUTE = 60
    HOUR = 3600
    DAY = 3600 * 24


class Duration:

    def __init__(self, duration: int, unit: TimeUnit) -> None:
        self._duration = duration
        self._unit = unit
        self._seconds = float(duration * unit.value)

    def seconds(self) -> float:
        return self._seconds

    def __str__(self) -> str:
        return f"{self._duration} {self._unit.name.lower()}(s)"

    @staticmethod
    def of(duration: int, unit: TimeUnit) -> "Duration":
        return Duration(duration, unit=unit)
