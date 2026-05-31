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
from typing import Any, Dict, List, Optional, Tuple

from .._typing import Number


class Parameter:
    __slots__ = ["name", "value", "types", "allow_empty", "possible_values"]

    def __init__(
        self,
        name: str,
        value: Any,
        types: Tuple,
        possible_values: Optional[List] = None,
        allow_empty: bool = True,
    ):
        self.name = name
        self.value = value
        self.types = types
        self.allow_empty = allow_empty
        self.possible_values = possible_values


def create_str_parameter(
    name: str,
    value: Any,
    possible_values: Optional[List[str]] = None,
    allow_empty: bool = True,
) -> Parameter:
    return Parameter(
        name=name,
        value=value,
        types=(str,),
        possible_values=possible_values,
        allow_empty=allow_empty,
    )


def create_int_parameter(
    name: str,
    value: Any,
    possible_values: Optional[List[int]] = None,
    allow_empty: bool = True,
) -> Parameter:
    return Parameter(
        name=name,
        value=value,
        types=(int,),
        possible_values=possible_values,
        allow_empty=allow_empty,
    )


def create_float_parameter(
    name: str,
    value: Any,
    possible_values: Optional[List[float]] = None,
    allow_empty: bool = True,
) -> Parameter:
    return Parameter(
        name=name,
        value=value,
        types=(float,),
        possible_values=possible_values,
        allow_empty=allow_empty,
    )


def create_numeric_parameter(
    name: str,
    value: Any,
    possible_values: Optional[List[Number]] = None,
    allow_empty: bool = True,
) -> Parameter:
    return Parameter(
        name=name,
        value=value,
        types=(float, int),
        possible_values=possible_values,
        allow_empty=allow_empty,
    )


def create_bool_parameter(
    name: str,
    value: Any,
    allow_empty: bool = True,
) -> Parameter:
    return Parameter(
        name=name,
        value=value,
        types=(bool,),
        possible_values=None,
        allow_empty=allow_empty,
    )


def create_list_parameter(
    name: str,
    value: Any,
    allow_empty: bool = True,
) -> Parameter:
    return Parameter(
        name=name,
        value=value,
        types=(list,),
        possible_values=None,
        allow_empty=allow_empty,
    )


def create_dict_parameter(
    name: str,
    value: Any,
    allow_empty: bool = True,
) -> Parameter:
    return Parameter(
        name=name,
        value=value,
        types=(dict,),
        possible_values=None,
        allow_empty=allow_empty,
    )
