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

from typing import Dict

from comet_ml.dataclasses import experiment_info


def _create_comet(
    experiment_info_: experiment_info.ExperimentInfo, share_api_key: bool
) -> Dict[str, str]:
    result = {
        "COMET_PROJECT_NAME": experiment_info_.project_name,
        "COMET_WORKSPACE": experiment_info_.workspace,
    }
    if share_api_key:
        result["COMET_API_KEY"] = experiment_info_.api_key

    result = {key: value for key, value in result.items() if value is not None}

    return result


def _create_vertex() -> Dict[str, str]:
    experiment_name = "{} - {}".format(
        "{{$.pipeline_task_name}}", "{{$.pipeline_job_name}}"
    )

    result = {
        "COMET_LOG_OTHER_VERTEX_RUN_NAME": "{{$.pipeline_job_name}}",
        "COMET_LOG_OTHER_VERTEX_TASK_NAME": "{{$.pipeline_task_name}}",
        "COMET_LOG_OTHER_VERTEX_TASK_ID": "{{$.pipeline_task_uuid}}",
        "COMET_LOG_OTHER_VERTEX_TASK_TYPE": "task",
        "COMET_LOG_OTHER_CREATED_FROM": "vertex",
        "COMET_EXPERIMENT_NAME": experiment_name,
    }

    return result


def create(
    experiment_info_: experiment_info.ExperimentInfo, share_api_key: bool
) -> Dict[str, str]:
    comet_environment = _create_comet(experiment_info_, share_api_key)
    vertex_environment = _create_vertex()

    environment = {**comet_environment, **vertex_environment}

    return environment
