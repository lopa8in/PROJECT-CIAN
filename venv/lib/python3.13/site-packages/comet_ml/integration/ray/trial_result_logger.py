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
from typing import Any, Dict

import box

from ...experiment import CometExperiment
from . import flatten_dictionary


def _is_valid_key_name(key, field):
    return key == field or key.startswith(field + "/")  # noqa: E731


class TrialResultLogger:
    _FIELD_CATEGORIES = {
        "not_logged": (
            "step",
            "epoch",
            "done",
            "should_checkpoint",
            "checkpoint_dir_name",
        ),
        "system": ("node_ip", "hostname", "pid", "date"),
        "episode": ("hist_stats/episode_reward", "hist_stats/episode_lengths"),
        "other": ("trial_id", "experiment_id", "experiment_tag"),
    }

    def __init__(self, experiment: CometExperiment, result: Dict[str, Any]) -> None:
        self._experiment = experiment
        self._result = result

    def process(self) -> None:
        step = self._result.get("step", 0)
        iteration = self._result.get("training_iteration", 1)
        epoch = self._result.get("epoch", iteration - 1)

        self._log_parameters(step)
        grouped = self._group_by_category()
        self._log_grouped_data(grouped, step=step, epoch=epoch)

    def _log_parameters(self, step: int) -> None:
        config_update = self._result.pop("config", {}).copy()
        config_update.pop("callbacks", None)  # Remove callbacks
        for key, value in config_update.items():
            if isinstance(value, dict):
                self._experiment.log_parameters(
                    flatten_dictionary.flatten({key: value}, "/"),
                    step=step,
                    nested_support=False,
                )
            else:
                self._experiment.log_parameter(key, value, step=step)

    def _group_by_category(self) -> box.Box:
        flattened = flatten_dictionary.flatten(self._result, delimiter="/")
        result = {category: {} for category in ["metric", "other", "system", "episode"]}

        for key, value in flattened.items():
            if value is None:
                continue
            category = self._category(key)
            if category == "not_logged":
                continue
            group = result[category]
            group[key] = value

        return box.Box(result)

    def _category(self, key: str) -> str:
        for category in "not_logged", "other", "system", "episode":
            if self._in_category(key, category):
                return category

        return "metric"

    def _in_category(self, key: str, category: str) -> bool:
        field_names = self._FIELD_CATEGORIES[category]
        return any(_is_valid_key_name(key, field) for field in field_names)

    def _log_grouped_data(self, grouped_data: box.Box, step: int, epoch: int) -> None:
        experiment = self._experiment
        experiment.log_others(grouped_data.other)
        experiment.log_metrics(grouped_data.metric, step=step, epoch=epoch)

        for key, value in grouped_data.system.items():
            experiment.log_system_info(key, value)

        for key, value in grouped_data.episode.items():
            experiment.log_curve(key, x=range(len(value)), y=value, step=step)
