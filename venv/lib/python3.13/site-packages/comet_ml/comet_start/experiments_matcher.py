# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.com
#  Copyright (C) 2015-2023 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
from typing import Any, Optional, Union

from comet_ml.logging_messages import (
    COMET_START_STARTING_NEW_EXPERIMENT,
    COMET_START_STARTING_NEW_EXPERIMENT_WITH_SOURCE,
)
from comet_ml.utils import masked_api_key

from .._online import Experiment
from ..constants import DEFAULT_PROJECT_NAME
from ..offline import OfflineExperiment
from .experiment_config import ExperimentConfig
from .init_parameters import InitParameters


class ExperimentMatchResult:
    def __init__(self, source: Optional[str]):
        self.source = source
        self.failures = []

    def add_failure(self, failure: str) -> None:
        self.failures.append(failure)

    def ok(self) -> bool:
        return len(self.failures) == 0

    def __bool__(self):
        return self.ok()

    def __str__(self) -> str:
        if self.ok():
            return "OK"

        if self.source is not None:
            message = COMET_START_STARTING_NEW_EXPERIMENT_WITH_SOURCE % self.source
        else:
            message = COMET_START_STARTING_NEW_EXPERIMENT

        for failure in self.failures:
            message = message + "\n" + failure

        return message


class MatchParameters:
    """
    Holds key matching parameters to be compared when deciding if specific experiment is the same as requested.
    """

    def __init__(
        self,
        api_key: Optional[str],
        workspace: Optional[str],
        project: Optional[str],
        experiment_key: Optional[str],
        online: Optional[bool],
        disabled: Optional[bool],
        offline_directory: Optional[str],
        distributed_node_identifier: Optional[str],
        source: Optional[str],
    ):
        self.api_key = api_key
        self.workspace = workspace
        self.project = project
        self.experiment_key = experiment_key
        self.online = online
        self.disabled = disabled
        self.distributed_node_identifier = distributed_node_identifier
        self.source = source
        if online is False:
            self.offline_directory = offline_directory
        else:
            self.offline_directory = None

    def __str__(self) -> str:
        return "MatchParameters: %r" % self.__dict__

    @staticmethod
    def build(
        experiment_config: ExperimentConfig, init_params: InitParameters
    ) -> "MatchParameters":
        offline_directory = None
        if (
            not init_params.online
            and not experiment_config.has_default_offline_directory()
        ):
            # use only non default offline directory for key parameters matching
            offline_directory = experiment_config.offline_directory

        # any parameter can be None - its expected - means user don't care about it
        return MatchParameters(
            api_key=init_params.api_key,
            workspace=init_params.workspace,
            project=init_params.project,
            experiment_key=init_params.experiment_key,
            online=init_params._online,
            disabled=experiment_config._disabled,
            offline_directory=offline_directory,
            distributed_node_identifier=experiment_config.distributed_node_identifier,
            source=init_params.source,
        )

    def match_experiment(
        self, experiment: Union[Experiment, OfflineExperiment]
    ) -> ExperimentMatchResult:
        offline_directory = None
        if isinstance(experiment, OfflineExperiment):
            offline_directory = experiment.offline_directory

        experiment_parameters = MatchParameters(
            api_key=experiment.api_key,
            workspace=experiment.workspace,
            project=experiment.project_name,
            experiment_key=experiment.get_key(),
            online=isinstance(experiment, Experiment),
            disabled=experiment.disabled,
            offline_directory=offline_directory,
            distributed_node_identifier=experiment.distributed_node_identifier,
            source=None,
        )

        result = ExperimentMatchResult(self.source)
        if not _none_or_equal(other=self.api_key, value=experiment_parameters.api_key):
            result.add_failure(
                f"api_key doesn't match ('{masked_api_key(self.api_key)}' != '{experiment_parameters.api_key}')"
            )

        if not _none_or_equal(
            other=self.workspace, value=experiment_parameters.workspace
        ):
            result.add_failure(
                f"workspace doesn't match ('{self.workspace}' != '{experiment_parameters.workspace}')"
            )

        if not _none_or_equal(
            other=self.project,
            value=experiment_parameters.project,
            default_value=DEFAULT_PROJECT_NAME,
        ):
            if self.project is None:
                project = DEFAULT_PROJECT_NAME
            else:
                project = self.project
            result.add_failure(
                f"project_name doesn't match ('{project}' != '{experiment_parameters.project}')"
            )

        if not _none_or_equal(
            other=self.experiment_key, value=experiment_parameters.experiment_key
        ):
            result.add_failure(
                f"experiment_key doesn't match ('{self.experiment_key}' != '{experiment_parameters.experiment_key}')"
            )

        if not _none_or_equal(other=self.online, value=experiment_parameters.online):
            result.add_failure(
                f"online doesn't match ({self.online} != {experiment_parameters.online})"
            )

        if not _none_or_equal(
            other=self.disabled, value=experiment_parameters.disabled
        ):
            result.add_failure(
                f"experiment_config.disabled doesn't match ({self.disabled} != {experiment_parameters.disabled})"
            )

        if not _none_or_equal(
            other=self.distributed_node_identifier,
            value=experiment_parameters.distributed_node_identifier,
        ):
            result.add_failure(
                f"experiment_config.distributed_node_identifier doesn't match "
                f"('{self.distributed_node_identifier}' != '{experiment_parameters.distributed_node_identifier}')"
            )

        if not _none_or_equal(
            other=self.offline_directory, value=experiment_parameters.offline_directory
        ):
            result.add_failure(
                f"experiment_config.offline_directory doesn't match "
                f"('{self.offline_directory}' != '{experiment_parameters.offline_directory}')"
            )

        return result


def _none_or_equal(value: Any, other: Any, default_value: Optional[Any] = None) -> bool:
    if other is None:
        return True

    # check if default value is matching other if provided
    # this can be used to match project_name which has default value "general" when not provided
    if value is None and default_value is not None:
        return default_value == other

    return value == other
