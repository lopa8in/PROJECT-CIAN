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
from typing import Any

from .helpers import validate_type_str
from .result import ValidationResult
from .validator import Validator


class CurveDataValidator(Validator):

    def __init__(self, name: str, x: Any, y: Any):
        self.name = name
        self.x = x
        self.y = y

    def validate(self) -> ValidationResult:
        res = ValidationResult(failed=False)
        if not validate_type_str(self.name, allow_empty=False):
            res.register_failures(
                ["'name' of the curve is mandatory and must be a string"]
            )

        if self.x is None or len(self.x) == 0:
            res.register_failures(["'x' values must be a list of numbers"])

        if self.y is None or len(self.y) == 0:
            res.register_failures(["'y' values must be a list of numbers"])

        if res.ok() and len(self.x) != len(self.y):
            res.register_failures(
                [
                    f"the number of 'x' values ({len(self.x)}) doesn't match the number of 'y' values ({len(self.y)})"
                ]
            )

        return res
