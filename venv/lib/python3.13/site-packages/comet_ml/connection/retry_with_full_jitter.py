# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2025 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************

import random

import urllib3


class RetryWithFullJitter(urllib3.Retry):
    """
    This class is a subclass of urllib3.Retry that adds full jitter to the backoff time.
    In the original base class jittering is done by adding a random value in pre-defined range [0, constant].
    In this class jittering is done by adding to backoff_time a random value in the range of [0, backoff_time].

    Additionally, this class is able to construct a Retry object respecting the urllib3 version, it handles
    correctly the backoff_max argument (didn't exist in urllib3 1.x), and the allowed_methods argument (removed in urllib3 2.x).
    """

    def __init__(self, *args, **kwargs) -> None:
        self._init_respecting_urllib3_version(*args, **kwargs)

    def _init_respecting_urllib3_version(self, *args, **kwargs) -> None:
        if urllib3.__version__.startswith("2."):
            super().__init__(*args, **kwargs)
            return

        backoff_max = kwargs.pop("backoff_max", self.DEFAULT_BACKOFF_MAX)
        self.DEFAULT_BACKOFF_MAX = backoff_max
        self.backoff_max = backoff_max

        if "allowed_methods" in kwargs:
            kwargs["method_whitelist"] = kwargs.pop("allowed_methods")

        super().__init__(*args, **kwargs)

    def get_backoff_time(self) -> float:
        backoff_time = super().get_backoff_time()
        backoff_time_with_full_jitter = backoff_time + backoff_time * random.random()
        return min(backoff_time_with_full_jitter, self.backoff_max)

    def new(self, **kwargs) -> "RetryWithFullJitter":
        result = super().new(**kwargs)

        result.backoff_max = self.backoff_max
        return result
