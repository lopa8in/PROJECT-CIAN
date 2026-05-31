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
import logging
from typing import Optional

from .._online import Experiment
from ..config import get_running_experiment
from ..constants import OTHER_KEY_CREATED_FROM
from ..experiment import CometExperiment
from ..logging_messages import (
    COMET_START_REUSING_EXPERIMENT_PROMPT,
    COMET_START_UNSUPPORTED_RUNNING_EXPERIMENT,
)
from ..offline import OfflineExperiment
from .experiment_config import ExperimentConfig
from .experiment_factory import (
    create_offline_experiment,
    create_online_experiment,
    resume_online_experiment,
)
from .experiments_matcher import MatchParameters
from .init_parameters import InitParameters

LOGGER = logging.getLogger(__name__)


class ConfigurationManager:
    def __init__(
        self,
        init_parameters: InitParameters,
        experiment_config: Optional[ExperimentConfig] = None,
    ):
        self.init_parameters = init_parameters
        if experiment_config is None:
            self.experiment_config = ExperimentConfig()
            LOGGER.debug(
                "User hasn't provided an experiment config, using default: %r",
                self.experiment_config,
            )
        else:
            self.experiment_config = experiment_config

    def validate(self):
        LOGGER.debug(
            "Validating configuration options provided by user. Initialization parameters: %s, experiment config: %s",
            self.init_parameters,
            self.experiment_config,
        )
        self.init_parameters.validate()

    def get_or_create_experiment(self) -> CometExperiment:
        experiment = self._get_or_create_experiment()
        if self.init_parameters.source is not None:
            experiment._log_other(
                OTHER_KEY_CREATED_FROM,
                self.init_parameters.source,
                include_context=False,
            )

        return experiment

    def _get_or_create_experiment(self) -> CometExperiment:
        self.validate()
        experiment = get_running_experiment()
        if experiment is not None and not experiment.ended:
            experiment = self.evaluate_running_experiment(experiment)
            if experiment is not None:
                LOGGER.info(COMET_START_REUSING_EXPERIMENT_PROMPT)
                return experiment

        if not self.init_parameters.online:
            return create_offline_experiment(
                experiment_config=self.experiment_config,
                init_parameters=self.init_parameters,
            )

        if self.init_parameters.experiment_key is None:
            return create_online_experiment(
                experiment_config=self.experiment_config,
                init_parameters=self.init_parameters,
            )
        else:
            return resume_online_experiment(
                experiment_config=self.experiment_config,
                init_parameters=self.init_parameters,
            )

    def evaluate_running_experiment(
        self, experiment: CometExperiment
    ) -> Optional[CometExperiment]:
        LOGGER.debug(
            "Running experiment found, evaluating whether to return it to the user"
        )
        if not isinstance(experiment, (Experiment, OfflineExperiment)):
            LOGGER.warning(COMET_START_UNSUPPORTED_RUNNING_EXPERIMENT)
            return None

        if self.init_parameters.is_get_or_create() or self.init_parameters.is_get():
            key_params = MatchParameters.build(
                experiment_config=self.experiment_config,
                init_params=self.init_parameters,
            )
            result = key_params.match_experiment(experiment)
            if result.ok():
                LOGGER.debug(
                    "Key parameters of the experiments have matched, so the existing experiment is returned"
                )
                return experiment
            else:
                LOGGER.warning(str(result))

        LOGGER.debug(
            "Since the key parameters of the experiments did not match, the current running experiment will be automatically terminated."
        )
        return None
