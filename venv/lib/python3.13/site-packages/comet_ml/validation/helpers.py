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
import numbers
from typing import Any, List, Optional, Tuple, Union

from .parameter import Parameter


def validate_type_str(value: Any, allow_empty: bool = True) -> bool:
    valid = validate_type(value, (str,), allow_empty=allow_empty)
    if not valid:
        return False

    if not allow_empty:
        return value != ""

    return True


def validate_type_int(value: Any, allow_empty: bool = True) -> bool:
    return validate_type(value, (int,), allow_empty=allow_empty)


def validate_type_float(value: Any, allow_empty: bool = True) -> bool:
    return validate_type(value, (float,), allow_empty=allow_empty)


def validate_type_numeric(value: Any, allow_empty: bool = True) -> bool:
    return validate_type(value, (float, int), allow_empty=allow_empty)


def validate_type_bool(value: Any, allow_empty: bool = True) -> bool:
    return validate_type(value, (bool,), allow_empty=allow_empty)


def validate_type_list(value: Any, allow_empty: bool = True) -> bool:
    return validate_type(value, (list,), allow_empty=allow_empty)


def validate_type_list_of_numbers(value: List[Any]) -> bool:
    if not validate_type_list(value, allow_empty=False):
        return False
    res = [validate_type_numeric(n, allow_empty=False) for n in value]
    return all(res)


def validate_type_dict(value: Any, allow_empty: bool = True) -> bool:
    return validate_type(value, (dict,), allow_empty=allow_empty)


def validate_type(value: Any, types: Tuple, allow_empty: bool) -> bool:
    if value is None:
        return allow_empty

    return any(isinstance(value, t) for t in types)


def validate_possible_values(
    value: Any, possible_values: Optional[List], allow_empty: bool
) -> bool:
    if value is None:
        return allow_empty

    return value in possible_values


def validate_value(value: Any, allow_empty: bool) -> bool:
    if value is None:
        return allow_empty

    if not allow_empty and hasattr(value, "__len__") and len(value) == 0:
        return False

    return True


def validate_parameter(parameter: Parameter) -> Tuple[bool, Optional[str]]:
    if parameter.possible_values is not None:
        valid_value = validate_possible_values(
            value=parameter.value,
            possible_values=parameter.possible_values,
            allow_empty=parameter.allow_empty,
        )
    else:
        valid_value = validate_value(
            value=parameter.value, allow_empty=parameter.allow_empty
        )

    if not valid_value:
        if parameter.possible_values is not None:
            possible_values_str = [str(v) for v in parameter.possible_values]

            if parameter.allow_empty:
                msg = "parameter %r must be one of [%s] or None but %r was given" % (
                    parameter.name,
                    ", ".join(possible_values_str),
                    parameter.value,
                )
            else:
                msg = "parameter %r must be one of [%s] but %r was given" % (
                    parameter.name,
                    ", ".join(possible_values_str),
                    parameter.value,
                )
        else:
            msg = "parameter %r must have non empty value but %r was given" % (
                parameter.name,
                parameter.value,
            )

        return False, msg

    valid_type = validate_type(
        value=parameter.value, types=parameter.types, allow_empty=parameter.allow_empty
    )

    if not valid_type:
        param_type = None if parameter.value is None else type(parameter.value).__name__

        if parameter.allow_empty:
            msg = "parameter %r must be of type(s) %r or None but %r was given" % (
                parameter.name,
                types_list(parameter.types),
                param_type,
            )
        else:
            msg = "parameter %r must be of type(s) %r but %r was given" % (
                parameter.name,
                types_list(parameter.types),
                param_type,
            )
        return False, msg

    return True, None


def types_list(types: Union[Tuple, List]) -> Union[str, List[str]]:
    type_names = []
    for t in types:
        type_names.append(t.__name__)

    if len(type_names) > 1:
        return type_names
    elif len(type_names) == 1:
        return type_names[0]
    else:
        return []


def flatten_list(xss: List[List[Any]]):
    return [x for xs in xss for x in xs]


def validate_box(item: Any) -> List[Union[int, float]]:
    """
    Validate a bounding box.
    """
    if len(item) != 4:
        raise ValueError("a box must be 4-tuple (x, y, width, height)")

    x, y, width, height = item
    if not isinstance(x, numbers.Number):
        raise ValueError("x must be a number")
    elif not isinstance(y, numbers.Number):
        raise ValueError("y must be a number")
    elif not isinstance(width, numbers.Number):
        raise ValueError("width must be a number")
    elif not isinstance(height, numbers.Number):
        raise ValueError("height must be a number")
    return [x, y, width, height]


def validate_polygon_points(item: Any) -> List[Tuple[Union[int, float]]]:
    """
    Validate a list of points for a polygon.
    """
    if not isinstance(item, (list, tuple)):
        raise ValueError("Polygon points must be a list or tuple")
    if len(item) < 3:
        raise ValueError("Polygon points must be composed of at least three points")
    for point in item:
        validate_point(point)

    points = flatten_list(item)
    return points


def validate_point(item: Any) -> None:
    """
    Validates an item as either an (x, y) point.
    """
    if not isinstance(item, (list, tuple)):
        raise ValueError("a Polygon point must be a list or tuple")
    if len(item) != 2:
        raise ValueError("a Polygon point must be composed of (x, y) components")

    x, y = item
    if not isinstance(x, numbers.Number):
        raise ValueError("a Polygon point's x must be a number")
    elif not isinstance(y, numbers.Number):
        raise ValueError("a Polygon point's y must be a number")
