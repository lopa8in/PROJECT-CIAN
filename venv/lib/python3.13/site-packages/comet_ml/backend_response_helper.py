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

from typing import Any, Dict, Optional, Tuple, Union

from comet_ml.convert_utils import try_convert_string_to_float

Timestamp = int
Value = Union[float, int, str, None]
ValueWithTimestamp = Tuple[Value, Timestamp]


def get_latest_metric_value(
    backend_response: Dict[str, Any]
) -> Optional[ValueWithTimestamp]:
    """
    Find the latest metric value in backend response from endpoint "experiment/metrics/get-metric"
    """

    if not ("metrics" in backend_response and len(backend_response["metrics"]) > 0):
        return None

    latest_value = backend_response["metrics"][-1]["metricValue"]
    latest_timestamp = backend_response["metrics"][-1]["timestamp"]

    # values in backend response are not sorted -> iterate over all values
    for metric in backend_response["metrics"]:
        # backend may return several metric values with the same timestamp
        # we will use value with max index (for that will use ">=" not just ">")
        if metric["timestamp"] >= latest_timestamp:
            latest_timestamp = metric["timestamp"]
            latest_value = metric["metricValue"]

    latest_value = try_convert_string_to_float(latest_value)

    return latest_value, latest_timestamp


def get_param_value(
    name: str,
    backend_response: Dict[str, Any],
) -> Optional[ValueWithTimestamp]:
    """
    Find parameter value in backend response from endpoint "experiment/parameters"
    """

    for value in backend_response["values"]:
        if value["name"] != name:
            continue

        param_value = value["valueCurrent"]
        param_timestamp = value["timestampCurrent"]

        param_value = try_convert_string_to_float(param_value)

        return param_value, param_timestamp

    return None


def get_other_value(
    name: str,
    backend_response: Dict[str, Any],
) -> Optional[ValueWithTimestamp]:
    """
    Find "other" value in backend response from endpoint "experiment/others"
    """

    return get_param_value(name, backend_response)


def is_backend_timestamp_newer(
    local_timestamp: Optional[Timestamp], backend_timestamp: Optional[Timestamp]
) -> bool:

    if backend_timestamp is None:
        return False
    if local_timestamp is None:
        return True

    return local_timestamp < backend_timestamp
