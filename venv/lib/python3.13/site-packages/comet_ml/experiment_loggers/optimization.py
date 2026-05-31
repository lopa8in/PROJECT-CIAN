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
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comet_ml import CometExperiment

from .._typing import Any, Dict, Optional


def log(
    experiment: "CometExperiment",
    optimization_id: Optional[str],
    metric_name: Optional[str],
    metric_value: Any,
    parameters: Optional[Dict],
    objective: Optional[str],
):
    if not experiment.alive:
        return None

    if optimization_id is not None:
        experiment.log_other("optimizer_id", optimization_id)

    if metric_name and metric_value:
        experiment.log_metric(metric_name, metric_value)

    if metric_name is not None:
        experiment.log_other("optimizer_metric", metric_name)
    if metric_value is not None:
        experiment.log_other("optimizer_metric_value", metric_value)

    if objective is not None:
        experiment.log_other("optimizer_objective", objective)

    if parameters is not None:
        experiment.log_parameters(parameters)

        try:
            json_parameters = json.dumps(parameters)
            experiment.log_other("optimizer_parameters", json_parameters)
        except Exception:
            pass
