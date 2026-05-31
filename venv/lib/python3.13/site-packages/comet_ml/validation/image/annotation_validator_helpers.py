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
from typing import Any, Dict, List

from ..helpers import (
    validate_type_dict,
    validate_type_list,
    validate_type_list_of_numbers,
    validate_type_numeric,
    validate_type_str,
)
from ..result import ValidationResult


def has_layer_names(annotations: List[Dict[str, Any]]) -> bool:
    names_exists = [
        "name" in a and a["name"] is not None and len(a["name"]) > 0
        for a in annotations
    ]
    return len(names_exists) > 0 and all(names_exists)


def has_data_fields(annotations: List[Dict[str, Any]]) -> bool:
    data_exists = [
        "data" in a and a["data"] is not None and len(a["data"]) > 0
        for a in annotations
    ]
    return len(data_exists) > 0 and all(data_exists)


def validate_boxes(boxes_data: Any) -> ValidationResult:
    res = None
    if validate_type_list(boxes_data, allow_empty=False) is False:
        res = ValidationResult(
            failure_reasons=[f"'boxes' must be a list, was provided: '{boxes_data}'"]
        )

    valid_values = [
        validate_type_list(b, allow_empty=False) and len(b) == 4 for b in boxes_data
    ]
    if len(valid_values) == 0 or not all(valid_values):
        res = ValidationResult(
            failure_reasons=[
                f"'boxes' must be a list of lists [[x, y, w, h], ...], was provided: '{boxes_data}'"
            ]
        )
    else:
        is_numbers = [validate_type_list_of_numbers(b) for b in boxes_data]
        if not all(is_numbers):
            res = ValidationResult(
                failure_reasons=[
                    f"'boxes' values must be a numbers, was provided: '{boxes_data}'"
                ]
            )

    if res is not None:
        return res

    return ValidationResult(failed=False)


def validate_points(points_data: Any) -> ValidationResult:
    res = None
    if validate_type_list(points_data, allow_empty=False) is False:
        res = ValidationResult(
            failure_reasons=[f"'points' must be a list, was provided: '{points_data}'"]
        )

    valid_values = [
        validate_type_list(b, allow_empty=False) and len(b) > 0 and len(b) % 2 == 0
        for b in points_data
    ]
    if len(valid_values) == 0 or not all(valid_values):
        res = ValidationResult(
            failure_reasons=[
                f"'points' must be a list of lists with points coordinates [[x1, y1, x2, y2, x3, y3 ...], ...], "
                f"was provided: '{points_data}'"
            ]
        )
    else:
        is_numbers = [validate_type_list_of_numbers(b) for b in points_data]
        if not all(is_numbers):
            res = ValidationResult(
                failure_reasons=[
                    f"'points' values must be a numbers, was provided: '{points_data}'"
                ]
            )

    if res is not None:
        return res

    return ValidationResult(failed=False)


def validate_data_section(data_section: Dict[str, Any]) -> ValidationResult:
    if len(data_section) == 0:
        return ValidationResult(
            failure_reasons=["'data' section is empty in one or all image annotations"]
        )

    if "boxes" in data_section:
        res = validate_boxes(data_section["boxes"])
    elif "points" in data_section:
        res = validate_points(data_section["points"])
    else:
        res = ValidationResult(
            failure_reasons=[
                f"'boxes' and/or 'points' must be in image annotation 'data' section: {data_section}"
            ]
        )

    if "label" not in data_section:
        res.register_failures(
            [f"'label' not found in image annotation 'data' section: {data_section}"]
        )

    if "score" in data_section and not validate_type_numeric(
        data_section["score"], allow_empty=True
    ):
        res.register_failures(
            [f"'score' value must be a number in 'data' section: {data_section}"]
        )

    if "id" in data_section and not validate_type_str(
        data_section["id"], allow_empty=False
    ):
        res.register_failures(
            [f"'id' value must be a string in 'data' section: {data_section}"]
        )

    if "metadata" in data_section and not validate_type_dict(
        data_section["metadata"], allow_empty=False
    ):
        res.register_failures(
            [f"'metadata' value must be a dictionary in 'data' section: {data_section}"]
        )

    return res
