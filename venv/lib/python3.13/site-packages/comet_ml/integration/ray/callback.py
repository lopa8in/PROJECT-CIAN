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
import logging
from typing import Any, Dict, List, Optional

import comet_ml
from comet_ml.dataclasses import hidden_api_key

import ray._private.worker as worker
import ray.tune.experiment
import ray.tune.logger

from ...constants import OTHER_KEY_CREATED_FROM
from . import callback_helpers, trial_result_logger, trial_save_logger

Trial = ray.tune.experiment.Trial

LOGGER = logging.getLogger(__name__)


class CometTrainLoggerCallback(ray.tune.logger.LoggerCallback):
    """
    Ray Callback for logging Train results to Comet.

    This Ray Train `LoggerCallback` sends metrics and parameters to
    Comet for tracking.

    This callback is based on the Ray native Comet callback and has been modified to allow to track
    resource usage on all distributed workers when running a distributed training job. It cannot be
    used with Ray Tune.

    Args:
        ray_config: Ray configuration dictionary to share with workers.
            It must be the same dictionary instance, not a copy.
        tags: Tags to add to the logged Experiment.
        save_checkpoints: If `True`, model checkpoints will be saved to
            Comet ML as artifacts.
        share_api_key_to_workers: If `True`, Comet API key will be shared
            with workers via ray_config dictionary. This is an unsafe solution and we recommend you
            uses a [more secure way to set up your API Key in your
            cluster](/docs/v2/guides/tracking-ml-training/distributed-training/).
        experiment_name: Custom name for the Comet experiment. If ``None``, a name
            is generated automatically.
        api_key (string): Comet API key.
        workspace (string): Comet workspace name.
        project_name (string): Comet project name.
        experiment_key (string): Experiment key to be used for logging.
        mode (string): Controls how the Comet experiment is started, 3 options are possible:

            - "get": Continue logging to an existing experiment identified by the `experiment_key` value.
            - "create": Always creates of a new experiment, useful for HPO sweeps.
            - "get_or_create" (default): Starts a fresh experiment if required, or persists logging to an existing one.
        online (bool): if True, the data will be logged to Comet server, otherwise it will be stored locally in offline experiment.
        experiment_kwargs: Other keyword arguments will be passed to the
            constructor for comet_ml.Experiment.

    Return: None

    Example:
        ```python linenums="1"
        config = {"lr": 1e-3, "batch_size": 64, "epochs": 20}

        comet_callback = CometTrainLoggerCallback(
            config,
            tags=["torch_ray_callback"],
            save_checkpoints=True,
            share_api_key_to_workers=True,
        )

        trainer = TorchTrainer(
            train_func,
            train_loop_config=config,
            scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
            run_config=RunConfig(callbacks=[comet_callback]),
        )
        result = trainer.fit()
        ```

    """

    def __init__(
        self,
        ray_config: Dict[str, Any],
        tags: Optional[List[str]] = None,
        save_checkpoints: bool = False,
        share_api_key_to_workers: bool = False,
        experiment_name: Optional[str] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        project_name: Optional[str] = None,
        experiment_key: Optional[str] = None,
        mode: Optional[str] = None,
        online: Optional[bool] = None,
        **experiment_kwargs  # fmt: skip
    ):
        self._save_checkpoints = save_checkpoints
        self._trial = None

        self._setup_shared_comet_experiment(
            api_key=api_key,
            workspace=workspace,
            project_name=project_name,
            experiment_key=experiment_key,
            mode=mode,
            online=online,
            tags=tags,
            experiment_name=experiment_name,
            **experiment_kwargs,
        )
        self._push_info_into_ray_configuration(ray_config, share_api_key_to_workers)

        if share_api_key_to_workers:
            LOGGER.warning(
                "Using CometTrainLoggerCallback(share_api_key_to_workers=True) is insecure. "
                "Please refer to https://www.comet.com/docs/v2/guides/tracking-ml-training/distributed-training/ for more secure alternatives."
            )

    @property
    def experiment_key(self):
        return self._experiment_key

    def _setup_shared_comet_experiment(
        self,
        api_key: Optional[str],
        workspace: Optional[str],
        project_name: Optional[str],
        experiment_key: Optional[str],
        mode: Optional[str],
        online: Optional[bool],
        tags: Optional[List[str]],
        experiment_name: Optional[str],
        **experiment_kwargs
    ):
        experiment_config_kwargs = {
            "log_env_gpu": False,
            "log_env_cpu": False,
            "log_env_disk": False,
            "log_env_network": False,
            "log_env_host": True,
            "display_summary_level": 0,
        }
        experiment_config_kwargs.update(experiment_kwargs)

        experiment_config = comet_ml.ExperimentConfig(**experiment_config_kwargs)

        experiment = comet_ml.start(
            api_key=api_key,
            workspace=workspace,
            project=project_name,
            experiment_key=experiment_key,
            mode=mode,
            online=online,
            experiment_config=experiment_config,
        )

        if experiment_name is not None:
            experiment.set_name(experiment_name)
        if tags is not None:
            experiment.add_tags(tags)
        experiment._log_other(OTHER_KEY_CREATED_FROM, "Ray", include_context=False)

        self._experiment_key = experiment.id
        self._api_key = experiment.api_key
        self._online = callback_helpers.is_online_experiment(experiment)

    def _push_info_into_ray_configuration(
        self, config: Dict[str, Any], share_api_key_to_workers: bool
    ):
        config["_comet_experiment_key"] = self._experiment_key
        if share_api_key_to_workers:
            config["_comet_api_key"] = hidden_api_key.HiddenApiKey(value=self._api_key)

        config["_comet_online"] = self._online

    def log_trial_start(self, trial: Trial):
        # new trial
        if self._trial is None:
            self._trial = trial
            self._setup_existing_shared_experiment(trial)
            return

        # already existing and started trial
        if trial.trial_id == self._trial.trial_id:
            return

        # trying to start new trial when there is one already running
        raise Exception(
            "CometTrainLoggerCallback has been already started. Only one start is allowed "
        )

    def _setup_existing_shared_experiment(self, trial):
        experiment_config = comet_ml.ExperimentConfig(
            log_env_gpu=False,
            log_env_cpu=False,
            log_env_details=False,
            log_env_host=False,
            display_summary_level=0,
        )

        # Depending on the Ray configuration(the usage of remote clusters), this experiment might be
        # started either in the same process where CometTrainLoggerCallback was instantiated
        # or in another one.
        experiment = comet_ml.start(
            experiment_key=self._experiment_key,
            api_key=self._api_key,
            experiment_config=experiment_config,
            online=self._online,
            mode="get",
        )

        config = trial.config.copy()
        config.pop("callbacks", None)
        if len(config) > 0:
            experiment.log_parameters(config, nested_support=False)

        self._experiment = experiment

    def log_trial_result(self, iteration: int, trial: Trial, result: Dict):
        if self._trial is None:
            self.log_trial_start(trial)

        if self._trial is not trial:
            raise Exception("Only one trial is allowed for CometTrainLoggerCallback")

        result_logger = trial_result_logger.TrialResultLogger(self._experiment, result)
        result_logger.process()

    def log_trial_save(self, trial: Trial):
        if self._save_checkpoints and trial.checkpoint is not None:
            trial_save_logger.go(self._experiment, trial)

    def log_trial_end(self, trial: Trial, failed: bool = False):
        # self._experiment.end()
        pass

    def on_experiment_end(self, trials: List["Trial"], **info):
        # if this is a worker mode -> end experiment manually to avoid losing any changes
        if worker.global_worker.mode == worker.WORKER_MODE:
            self._experiment.end()
