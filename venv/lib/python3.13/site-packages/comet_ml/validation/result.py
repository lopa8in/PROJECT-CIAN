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
from typing import List, Optional


class ValidationResult:
    def __init__(
        self, failed: bool = True, failure_reasons: Optional[List[str]] = None
    ):
        self._failed = failed
        self._failure_reasons = failure_reasons

    def failed(self):
        return self._failed

    def ok(self):
        return not self._failed

    @property
    def failure_reasons(self) -> List[str]:
        return self._failure_reasons

    def register_failures(self, failures: List[str]) -> None:
        if self._failure_reasons is None:
            self._failure_reasons = failures
        else:
            self._failure_reasons.extend(failures)
        self._failed = True

    def __bool__(self):
        return not self._failed

    def __str__(self) -> str:
        if not self._failed:
            return "OK"

        return f"Failed: {self._failure_reasons}"
