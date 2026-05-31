# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2021 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
from ..types import Event, Hint
from .handler import Handler


class FilterCountHandler(Handler):
    def __init__(self, max_count: int, level: str):
        super().__init__()
        self._count = 0
        self._level = level
        self._max_count = max_count

    def process_event(self, event: Event, hint: Hint) -> bool:
        if not event.get("level", "") == self._level:
            return True

        if self._count >= self._max_count:
            return False

        self._count += 1
        return True
