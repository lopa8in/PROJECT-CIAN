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
import logging
from collections.abc import Iterable
from typing import Any, List, Optional

from comet_ml.validation.helpers import validate_parameter
from comet_ml.validation.parameter import (
    Parameter,
    create_bool_parameter,
    create_dict_parameter,
    create_float_parameter,
    create_int_parameter,
    create_list_parameter,
    create_numeric_parameter,
    create_str_parameter,
)
from comet_ml.validation.result import ValidationResult
from comet_ml.validation.validator import Validator

from .._typing import Number
from .error import ValidationError


class MethodParametersTypeValidator(Validator):
    def __init__(self, method_name: str, class_name: Optional[str] = None):
        if class_name is not None:
            self.prefix: str = "%s.%s" % (class_name, method_name)
        else:
            self.prefix: str = method_name
        self.parameters: List[Parameter] = []
        self.validation_result: Optional[ValidationResult] = None

    def add_str_parameter(
        self,
        value: Any,
        name: str,
        possible_values: Optional[List[str]] = None,
        allow_empty: bool = True,
    ):
        self.parameters.append(
            create_str_parameter(
                name=name,
                value=value,
                possible_values=possible_values,
                allow_empty=allow_empty,
            )
        )

    def add_bool_parameter(self, value: Any, name: str, allow_empty: bool = True):
        self.parameters.append(
            create_bool_parameter(name=name, value=value, allow_empty=allow_empty)
        )

    def add_int_parameter(
        self,
        value: Any,
        name: str,
        possible_values: Optional[List[int]] = None,
        allow_empty: bool = True,
    ):
        self.parameters.append(
            create_int_parameter(
                name=name,
                value=value,
                possible_values=possible_values,
                allow_empty=allow_empty,
            )
        )

    def add_float_parameter(
        self,
        value: Any,
        name: str,
        possible_values: Optional[List[float]] = None,
        allow_empty: bool = True,
    ):
        self.parameters.append(
            create_float_parameter(
                name=name,
                value=value,
                possible_values=possible_values,
                allow_empty=allow_empty,
            )
        )

    def add_numeric_parameter(
        self,
        value: Any,
        name: str,
        possible_values: Optional[List[Number]] = None,
        allow_empty: bool = True,
    ):
        self.parameters.append(
            create_numeric_parameter(
                name=name,
                value=value,
                possible_values=possible_values,
                allow_empty=allow_empty,
            )
        )

    def add_list_parameter(self, value: Any, name: str, allow_empty: bool = True):
        self.parameters.append(
            create_list_parameter(name=name, value=value, allow_empty=allow_empty)
        )

    def add_list_with_strings_parameter(
        self, value: Any, name: str, allow_empty: bool = True
    ):
        self.parameters.append(
            create_list_parameter(name=name, value=value, allow_empty=allow_empty)
        )
        if isinstance(value, Iterable) and not isinstance(value, str):
            for ix, string in enumerate(value):
                name_ = f"{name}[{ix}]"
                self.parameters.append(
                    create_str_parameter(
                        name=name_, value=string, allow_empty=allow_empty
                    )
                )

    def add_dict_parameter(self, value: Any, name: str, allow_empty: bool = True):
        self.parameters.append(
            create_dict_parameter(name=name, value=value, allow_empty=allow_empty)
        )

    def validate(self) -> ValidationResult:
        try:
            failures = []
            for parameter in self.parameters:
                valid, msg = validate_parameter(parameter)
                if not valid:
                    failures.append(msg)

            if len(failures) > 0:
                self.validation_result = ValidationResult(failure_reasons=failures)
            else:
                self.validation_result = ValidationResult(failed=False)
        except Exception as e:
            self.validation_result = ValidationResult(
                failure_reasons=["Unexpected validation error: %r" % e]
            )

        return self.validation_result

    def print_result(self, logger: logging.Logger, log_level: int = logging.ERROR):
        if self.validation_result is None:
            logger.log(
                level=log_level, msg="No validation result, please call validate first"
            )
            return

        for msg in self.validation_result.failure_reasons:
            logger.log(log_level, "%s: %s", self.prefix, msg)

    def throw_validation_error(self):
        if (
            self.validation_result is not None
            and len(self.validation_result.failure_reasons) > 0
        ):
            raise ValidationError(
                failure_reasons=self.validation_result.failure_reasons
            )
