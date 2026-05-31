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
from .duration import Duration


class OpRate:
    """Operation rate definition
    Args:
        op_limit: number of operations allowed within `interval`
        interval: time interval
    """

    def __init__(self, op_limit: int, interval: Duration):
        self._op_limit = op_limit
        self._interval = interval

    @property
    def op_limit(self):
        return self._op_limit

    @property
    def interval_seconds(self) -> float:
        return self._interval.seconds()

    def __str__(self) -> str:
        return f"{self._op_limit} per {self._interval}"
