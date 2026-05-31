# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.com
#  Copyright (C) 2015-2024 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
import logging
from typing import Optional

from ..experiment import CometExperiment
from ..logging_messages import COMET_START_FAILED_TO_CREATE_EXPERIMENT_ERROR
from .config_manager import ConfigurationManager
from .experiment_config import ExperimentConfig
from .init_parameters import InitParameters

ExperimentBase = CometExperiment

LOGGER = logging.getLogger(__name__)


def start(
    api_key: Optional[str] = None,
    workspace: Optional[str] = None,
    project: Optional[str] = None,
    project_name: Optional[str] = None,
    experiment_key: Optional[str] = None,
    mode: Optional[str] = None,
    online: Optional[bool] = None,
    source: Optional[str] = None,
    experiment_config: Optional[ExperimentConfig] = None,
) -> CometExperiment:
    """
    Returns an Experiment object to log data to. It can be used to log data to a new Comet Experiment, append data to an
    Existing Experiment or continue logging to a running experiment locally.

    Args:
        api_key (str, optional): Comet API key. It's recommended to configure the API Key with `comet login`.
        workspace (str, optional): Comet workspace name. If not provided, uses the default workspace.
        project (str, optional): Comet project name. [Deprecated, use ``project_name`` instead]
        project_name (str, optional): Comet project name. Defaults to `Uncategorized`.
        experiment_key (str, optional): The Experiment identifier to be used for logging. This is used either to append
            data to an Existing Experiment or to control the key of new experiments (for example to match another
            identifier). Must be an alphanumeric string whose length is between 32 and 50 characters.
        mode (str, optional): Control how the Comet experiment is started.

            * ``"get_or_create"``: Starts a fresh experiment if required, or persists logging to an existing one.
            * ``"get"``: Continue logging to an existing experiment identified by the ``experiment_key`` value.
            * ``"create"``: Always creates of a new experiment, useful for HPO sweeps.
        online (boolean, optional): If True, the data will be logged to Comet server, otherwise it will be stored locally
            in an offline experiment. Default is ``True``.
        source (str, optional): Enables the specification of the origin of an experiment.
        experiment_config (Optional[ExperimentConfig], optional): additional configuration options.

    Returns:
        The initialized Comet Experiment object.

    Raises:
        ValueError: If no API Key is set in online mode.
        ExperimentNotFound: If mode="get" and the experiment_key doesn't
            exist, or you don't have access to it.
        InvalidExperimentMode:

            * If mode="get" but no experiment_key was passed or configured.
            * If mode="create", an experiment_key was passed or configured and
              an Experiment with that Key already exists.


    """
    if project_name is None:
        project_name = project

    init_parameters = InitParameters(
        api_key=api_key,
        workspace=workspace,
        project_name=project_name,
        experiment_key=experiment_key,
        mode=mode,
        online=online,
        source=source,
    )
    config_manager = ConfigurationManager(
        init_parameters=init_parameters, experiment_config=experiment_config
    )
    try:
        return config_manager.get_or_create_experiment()
    except Exception as e:
        LOGGER.error(COMET_START_FAILED_TO_CREATE_EXPERIMENT_ERROR, e, exc_info=True)
        raise e
