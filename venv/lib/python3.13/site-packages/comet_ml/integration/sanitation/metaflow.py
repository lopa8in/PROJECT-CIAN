# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.com
#  Copyright (C) 2015-2021 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************

from comet_ml._typing import Any, Dict


def _clear_metaflow_vars(pipeline_details: Dict[str, Any]) -> None:
    if "attributes" not in pipeline_details:
        return
    if "vars" not in pipeline_details["attributes"]:
        return

    pipeline_details["attributes"]["vars"] = {}


def sanitize_pipeline_environment(pipeline_details: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(pipeline_details, dict):
        return pipeline_details

    for key, value in pipeline_details.items():
        if value == "environment":
            _clear_metaflow_vars(pipeline_details)
        if isinstance(value, list):
            for item in value:
                sanitize_pipeline_environment(item)

        sanitize_pipeline_environment(value)
    return pipeline_details
