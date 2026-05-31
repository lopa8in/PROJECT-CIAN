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

import json
import math
import traceback
from inspect import istraceback
from typing import Any

convert_functions = []

try:
    import numpy

    def convert_numpy_array_pre_1_16(value: Any) -> Any:
        try:
            return numpy.asscalar(value)
        except (ValueError, IndexError, AttributeError, TypeError):
            return

    def convert_numpy_array_post_1_16(value: Any) -> Any:
        try:
            return value.item()
        except (ValueError, IndexError, AttributeError, TypeError):
            return

    convert_functions.append(convert_numpy_array_post_1_16)
    convert_functions.append(convert_numpy_array_pre_1_16)
except ImportError:
    pass


def nan2None(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: nan2None(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [nan2None(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


class NestedEncoder(json.JSONEncoder):
    """
    A JSON Encoder that converts floats/decimals to strings and allows nested objects
    """

    def encode(self, obj: Any, *args, **kwargs) -> Any:
        obj = nan2None(obj)
        return super().encode(obj, *args, **kwargs)

    def iterencode(self, obj: Any, *args, **kwargs) -> Any:
        obj = nan2None(obj)
        return super().iterencode(obj, *args, **kwargs)

    def default(self, obj: Any) -> Any:

        # raise TypeError("test")

        # First convert the object
        obj = NestedEncoder.convert(obj)

        # Check if the object is convertible
        try:
            json.JSONEncoder().encode(obj)
            return obj

        except TypeError:
            pass

        # Custom conversion
        if isinstance(obj, (Exception, type)):
            return str(obj)

        elif istraceback(obj):
            return "".join(traceback.format_tb(obj)).strip()

        elif hasattr(obj, "repr_json"):
            return obj.repr_json()

        elif isinstance(obj, complex):
            return str(obj)

        else:
            try:
                return json.JSONEncoder.default(self, obj)

            except TypeError:
                return "%s is not JSON serializable" % obj.__class__.__name__

    @staticmethod
    def convert(obj):
        """
        Try converting the obj to something json-encodable
        """
        for converter in convert_functions:
            converted = converter(obj)

            if converted is not None:
                obj = converted

        return obj


class NestedNumpyOrTensorEncoder(json.JSONEncoder):
    def encode(self, obj: Any) -> Any:
        obj = nan2None(obj)
        return super().encode(obj)

    def iterencode(self, obj: Any, *args: Any, **kwargs: Any):
        obj = nan2None(obj)
        return super().iterencode(obj, *args, **kwargs)

    def default(self, obj: Any) -> Any:
        # First convert the object to something that can be encoded into JSON
        obj = NestedNumpyOrTensorEncoder.convert(obj)

        # Check if the object is convertible
        try:
            json.JSONEncoder().encode(obj)
            return obj

        except TypeError:
            pass

        # Custom conversion
        if isinstance(obj, (Exception, type)):
            return str(obj)
        elif isinstance(obj, complex):
            return str(obj)
        else:
            return json.JSONEncoder.default(self, obj)

    @staticmethod
    def convert(obj: Any) -> Any:
        # first try to convert to list from Numpy NDArray or Tensor
        try:
            from comet_ml.convert_utils import convert_to_list

            return convert_to_list(obj, multidimensional=True)
        except Exception:
            pass

        # try to convert using Numpy converters for arrays of size 1
        for converter in convert_functions:
            converted = converter(obj)

            if converted is not None:
                obj = converted

        return obj
