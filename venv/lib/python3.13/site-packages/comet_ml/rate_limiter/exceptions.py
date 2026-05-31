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
from typing import Dict, Union

from .rate import OpRate


class LimitRateExceededError(Exception):

    def __init__(self, identity: str, rate: OpRate, remaining_time: float):
        error = f"Limit rate of {rate} for '{identity}' has been exceeded"
        self.meta_data: Dict[str, Union[str, float]] = {
            "identity": identity,
            "rate": str(rate),
            "remaining_time": remaining_time,
        }
        super().__init__(error)
