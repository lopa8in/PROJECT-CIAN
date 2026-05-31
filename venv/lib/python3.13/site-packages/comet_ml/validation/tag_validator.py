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
from typing import List, Optional

from comet_ml.validation.method_parameters_validator import (
    MethodParametersTypeValidator,
)
from comet_ml.validation.result import ValidationResult
from comet_ml.validation.validator import Validator

LOGGER = logging.getLogger(__name__)


class TagValidator(Validator):
    """
    Validator for tag (just one value)
    """

    def __init__(self, tag: str, method_name: str, class_name: Optional[str] = None):
        self.param_validator = MethodParametersTypeValidator(
            method_name=method_name, class_name=class_name
        )
        self.tag = tag
        self.validation_result: Optional[ValidationResult] = None

    def validate(self) -> ValidationResult:
        try:
            self.param_validator.add_str_parameter(self.tag, "tag", allow_empty=False)
            self.validation_result = self.param_validator.validate()

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
            logger.log(log_level, "%s: %s", self.param_validator.prefix, msg)


class TagsValidator(Validator):
    """
    Validator for tags (list of tags)
    """

    def __init__(
        self, tags: List[str], method_name: str, class_name: Optional[str] = None
    ):
        self.param_validator = MethodParametersTypeValidator(
            method_name=method_name, class_name=class_name
        )
        self.tags = tags
        self.validation_result: Optional[ValidationResult] = None

    def validate(self) -> ValidationResult:
        try:
            self.param_validator.add_list_with_strings_parameter(
                self.tags, "tags", allow_empty=False
            )
            self.validation_result = self.param_validator.validate()

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
            logger.log(log_level, "%s: %s", self.param_validator.prefix, msg)
