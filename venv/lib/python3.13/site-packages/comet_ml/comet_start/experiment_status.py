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
import logging

import comet_ml.api

from ..connection.connection_factory import get_rest_server_connection
from ..exceptions import ExperimentAlreadyUploaded
from .init_parameters import InitParameters

LOGGER = logging.getLogger(__name__)


def check_experiment_already_exists(init_parameters: InitParameters) -> bool:
    if init_parameters.is_get_or_create_with_experiment_key():
        return _experiment_exists_by_add_run(init_parameters)
    else:
        return _experiment_exists_by_metadata(init_parameters)


def _experiment_exists_by_metadata(init_parameters: InitParameters) -> bool:
    api = comet_ml.api.get_instance(api_key=init_parameters.api_key, cache=False)

    experiment = api.get_experiment_by_key(
        experiment_key=init_parameters.experiment_key
    )
    return experiment is not None


def _experiment_exists_by_add_run(init_parameters: InitParameters) -> bool:
    api_client = get_rest_server_connection(
        api_key=init_parameters.api_key,
        experiment_key=init_parameters.experiment_key,
    )
    try:
        response = api_client.add_run(
            project_name=init_parameters.project,
            workspace=init_parameters.workspace,
            offline=init_parameters.offline,
            get_or_create_mode=False,
        )
        LOGGER.debug("Does the experiment exists by add_run? Response: %r", response)
    except ExperimentAlreadyUploaded:
        return True

    return False
