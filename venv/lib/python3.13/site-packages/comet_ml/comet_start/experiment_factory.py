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

from .._online import ExistingExperiment, Experiment
from ..constants import (
    RESUME_STRATEGY_CREATE,
    RESUME_STRATEGY_GET,
    RESUME_STRATEGY_GET_OR_CREATE,
)
from ..exceptions import ExperimentNotFound, InvalidExperimentMode
from ..experiment import CometExperiment
from ..logging_messages import (
    COMET_START_EXPERIMENT_CAN_NOT_BE_CREATED_ALREADY_EXISTS_EXCEPTION,
    COMET_START_EXPERIMENT_CAN_NOT_BE_CREATED_ERROR,
    COMET_START_EXPERIMENT_WITH_KEY_NOT_FOUND_ERROR,
    COMET_START_EXPERIMENT_WITH_KEY_NOT_FOUND_EXCEPTION,
    COMET_START_OFFLINE_GET_WITHOUT_EXPERIMENT_KEY_ERROR,
    COMET_START_OFFLINE_GET_WITHOUT_EXPERIMENT_KEY_EXCEPTION,
    COMET_START_ONLINE_GET_WITHOUT_EXPERIMENT_KEY_ERROR,
    COMET_START_ONLINE_GET_WITHOUT_EXPERIMENT_KEY_EXCEPTION,
    COMET_START_UNSUPPORTED_EXPERIMENT_CREATION_MODE_ERROR,
    COMET_START_UNSUPPORTED_EXPERIMENT_CREATION_MODE_EXCEPTION,
)
from ..offline import OfflineExperiment
from .experiment_config import ExperimentConfig, effective_experiment_config
from .experiment_status import check_experiment_already_exists
from .init_parameters import InitParameters

LOGGER = logging.getLogger(__name__)


def resume_online_experiment(
    experiment_config: ExperimentConfig, init_parameters: InitParameters
) -> CometExperiment:
    exist_in_backend = check_experiment_already_exists(init_parameters)

    if init_parameters.mode == RESUME_STRATEGY_CREATE:
        new_experiment = True
    elif not exist_in_backend and init_parameters.mode == RESUME_STRATEGY_GET_OR_CREATE:
        new_experiment = True
    else:
        new_experiment = False

    # create effective configuration
    effective_config = effective_experiment_config(
        user_config=experiment_config, new_experiment=new_experiment
    )

    LOGGER.debug(
        "Resuming/creating online experiment with effective experiment configuration: %r, "
        "init params: %r, exists in backend: %r",
        effective_config,
        init_parameters,
        exist_in_backend,
    )

    if init_parameters.mode == RESUME_STRATEGY_GET:
        if exist_in_backend:
            use_existing = True
        else:
            LOGGER.error(
                COMET_START_EXPERIMENT_WITH_KEY_NOT_FOUND_ERROR,
                init_parameters.experiment_key,
                init_parameters,
                effective_config,
            )
            raise ExperimentNotFound(
                COMET_START_EXPERIMENT_WITH_KEY_NOT_FOUND_EXCEPTION
                % init_parameters.experiment_key
            )
    elif init_parameters.mode == RESUME_STRATEGY_GET_OR_CREATE:
        use_existing = exist_in_backend
    elif init_parameters.mode == RESUME_STRATEGY_CREATE:
        if exist_in_backend:
            LOGGER.error(
                COMET_START_EXPERIMENT_CAN_NOT_BE_CREATED_ERROR,
                init_parameters.experiment_key,
                init_parameters,
                effective_config,
            )
            raise InvalidExperimentMode(
                COMET_START_EXPERIMENT_CAN_NOT_BE_CREATED_ALREADY_EXISTS_EXCEPTION
                % (
                    init_parameters.experiment_key,
                    RESUME_STRATEGY_GET,
                    RESUME_STRATEGY_GET_OR_CREATE,
                )
            )
        else:
            use_existing = False
    else:
        LOGGER.error(
            COMET_START_UNSUPPORTED_EXPERIMENT_CREATION_MODE_ERROR,
            init_parameters.mode,
            init_parameters,
            effective_config,
        )
        raise InvalidExperimentMode(
            COMET_START_UNSUPPORTED_EXPERIMENT_CREATION_MODE_EXCEPTION
            % init_parameters.mode
        )

    return _create_online_experiment(
        experiment_config=effective_config,
        init_parameters=init_parameters,
        existing=use_existing,
    )


def create_online_experiment(
    experiment_config: ExperimentConfig, init_parameters: InitParameters
) -> CometExperiment:
    # create effective configuration
    effective_config = effective_experiment_config(
        user_config=experiment_config, new_experiment=True
    )
    LOGGER.debug(
        "Creating online experiment with effective experiment configuration: %r, init params: %r",
        effective_config,
        init_parameters,
    )

    if init_parameters.mode == RESUME_STRATEGY_GET:
        LOGGER.error(
            COMET_START_ONLINE_GET_WITHOUT_EXPERIMENT_KEY_ERROR,
            init_parameters,
            effective_config,
        )
        raise InvalidExperimentMode(
            COMET_START_ONLINE_GET_WITHOUT_EXPERIMENT_KEY_EXCEPTION
        )

    return _create_online_experiment(
        experiment_config=effective_config,
        init_parameters=init_parameters,
        existing=False,
    )


def create_offline_experiment(
    experiment_config: ExperimentConfig, init_parameters: InitParameters
) -> CometExperiment:
    # Try to estimate what kind of configuration options should be applied to created offline experiment.
    # Here we can only rely on the experiment creation mode because we can not check against backend if
    # the experiment exists of not. Thus, the most conservative approach applied - both create modes assumed
    # to have configuration options as for new experiment.
    if (
        init_parameters.mode == RESUME_STRATEGY_CREATE
        or init_parameters.mode == RESUME_STRATEGY_GET_OR_CREATE
    ):
        new_experiment = True
    else:
        new_experiment = False

    # create effective configuration
    effective_config = effective_experiment_config(
        user_config=experiment_config, new_experiment=new_experiment
    )
    LOGGER.debug(
        "Creating offline experiment with effective experiment configuration: %r, init params: %r",
        effective_config,
        init_parameters,
    )

    if init_parameters.experiment_key is None:
        if init_parameters.is_get():
            LOGGER.error(
                COMET_START_OFFLINE_GET_WITHOUT_EXPERIMENT_KEY_ERROR,
                init_parameters,
                experiment_config,
            )
            raise InvalidExperimentMode(
                COMET_START_OFFLINE_GET_WITHOUT_EXPERIMENT_KEY_EXCEPTION
            )

    # We are forced to create here full OfflineExperiment because until attempt to upload it to the server there is
    # no way to know if it is new experiment of already existing. Thus, in order to properly configure experiment
    # object with parameters that exists in OfflineExperiment constructor but missing in ExistingOfflineExperiment
    # constructor we do fallback to the most wide option available.
    offline_experiment = OfflineExperiment(
        project_name=init_parameters.project,
        workspace=init_parameters.workspace,
        log_code=effective_config.log_code,
        log_graph=effective_config.log_graph,
        auto_param_logging=effective_config.auto_param_logging,
        auto_metric_logging=effective_config.auto_metric_logging,
        parse_args=effective_config.parse_args,
        auto_output_logging=effective_config.auto_output_logging,
        log_env_details=effective_config.log_env_details,
        log_git_metadata=effective_config.log_git_metadata,
        log_git_patch=effective_config.log_git_patch,
        disabled=effective_config.disabled,
        offline_directory=effective_config.offline_directory,
        log_env_gpu=effective_config.log_env_gpu,
        log_env_host=effective_config.log_env_host,
        log_env_cpu=effective_config.log_env_cpu,
        log_env_network=effective_config.log_env_network,
        log_env_disk=effective_config.log_env_disk,
        display_summary_level=effective_config.display_summary_level,
        auto_log_co2=effective_config.auto_log_co2,
        auto_metric_step_rate=effective_config.auto_metric_step_rate,
        auto_histogram_tensorboard_logging=effective_config.auto_histogram_tensorboard_logging,
        auto_histogram_weight_logging=experiment_config.auto_histogram_weight_logging,
        auto_histogram_epoch_rate=effective_config.auto_histogram_epoch_rate,
        auto_histogram_gradient_logging=effective_config.auto_histogram_gradient_logging,
        auto_histogram_activation_logging=effective_config.auto_histogram_activation_logging,
        experiment_key=init_parameters.experiment_key,
        distributed_node_identifier=experiment_config.distributed_node_identifier,
        _resume_strategy=init_parameters.mode,
        _comet_start_sourced=True,
    )
    if experiment_config.name is not None:
        offline_experiment.set_name(experiment_config.name)
    if experiment_config.tags is not None and len(experiment_config.tags) > 0:
        offline_experiment.add_tags(experiment_config.tags)

    return offline_experiment


def _create_online_experiment(
    experiment_config: ExperimentConfig, init_parameters: InitParameters, existing: bool
) -> CometExperiment:
    if not existing:
        LOGGER.debug("Creating online experiment: %r", init_parameters)
        experiment = Experiment(
            api_key=init_parameters.api_key,
            project_name=init_parameters.project,
            workspace=init_parameters.workspace,
            log_code=experiment_config.log_code,
            log_graph=experiment_config.log_graph,
            auto_param_logging=experiment_config.auto_param_logging,
            auto_metric_logging=experiment_config.auto_metric_logging,
            parse_args=experiment_config.parse_args,
            auto_output_logging=experiment_config.auto_output_logging,
            log_env_details=experiment_config.log_env_details,
            log_git_metadata=experiment_config.log_git_metadata,
            log_git_patch=experiment_config.log_git_patch,
            disabled=experiment_config.disabled,
            log_env_gpu=experiment_config.log_env_gpu,
            log_env_host=experiment_config.log_env_host,
            log_env_cpu=experiment_config.log_env_cpu,
            log_env_network=experiment_config.log_env_network,
            log_env_disk=experiment_config.log_env_disk,
            display_summary_level=experiment_config.display_summary_level,
            auto_log_co2=experiment_config.auto_log_co2,
            auto_metric_step_rate=experiment_config.auto_metric_step_rate,
            auto_histogram_tensorboard_logging=experiment_config.auto_histogram_tensorboard_logging,
            auto_histogram_weight_logging=experiment_config.auto_histogram_weight_logging,
            auto_histogram_epoch_rate=experiment_config.auto_histogram_epoch_rate,
            auto_histogram_gradient_logging=experiment_config.auto_histogram_gradient_logging,
            auto_histogram_activation_logging=experiment_config.auto_histogram_activation_logging,
            experiment_key=init_parameters.experiment_key,
            distributed_node_identifier=experiment_config.distributed_node_identifier,
            enable_get_or_create_mode=init_parameters.is_get_or_create_with_experiment_key(),
            offline_directory=experiment_config.offline_directory,
        )
    else:
        LOGGER.debug("Creating existing online experiment: %r", init_parameters)
        experiment = ExistingExperiment(
            api_key=init_parameters.api_key,
            project_name=init_parameters.project,
            workspace=init_parameters.workspace,
            log_code=experiment_config.log_code,
            log_graph=experiment_config.log_graph,
            auto_param_logging=experiment_config.auto_param_logging,
            auto_metric_logging=experiment_config.auto_metric_logging,
            parse_args=experiment_config.parse_args,
            auto_output_logging=experiment_config.auto_output_logging,
            log_env_details=experiment_config.log_env_details,
            log_git_metadata=experiment_config.log_git_metadata,
            log_git_patch=experiment_config.log_git_patch,
            disabled=experiment_config.disabled,
            log_env_gpu=experiment_config.log_env_gpu,
            log_env_host=experiment_config.log_env_host,
            log_env_cpu=experiment_config.log_env_cpu,
            log_env_network=experiment_config.log_env_network,
            log_env_disk=experiment_config.log_env_disk,
            display_summary_level=experiment_config.display_summary_level,
            auto_log_co2=experiment_config.auto_log_co2,
            auto_metric_step_rate=experiment_config.auto_metric_step_rate,
            auto_histogram_tensorboard_logging=experiment_config.auto_histogram_tensorboard_logging,
            auto_histogram_weight_logging=experiment_config.auto_histogram_weight_logging,
            auto_histogram_epoch_rate=experiment_config.auto_histogram_epoch_rate,
            auto_histogram_gradient_logging=experiment_config.auto_histogram_gradient_logging,
            auto_histogram_activation_logging=experiment_config.auto_histogram_activation_logging,
            experiment_key=init_parameters.experiment_key,
            distributed_node_identifier=experiment_config.distributed_node_identifier,
            offline_directory=experiment_config.offline_directory,
        )

    if experiment_config.name is not None:
        experiment.set_name(experiment_config.name)
    if experiment_config.tags is not None and len(experiment_config.tags) > 0:
        experiment.add_tags(experiment_config.tags)

    return experiment
