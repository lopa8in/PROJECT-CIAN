# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.com
#  Copyright (C) 2015-2025 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
from typing import Any, List, Optional

from ray.train.data_parallel_trainer import DataParallelTrainer

from .callback import CometTrainLoggerCallback


def comet_ray_train_logger(
    trainer: DataParallelTrainer,
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
) -> None:
    """
    Enables the registration of a Comet Ray callback with the specified trainer to collect and
    send training metrics and parameters to Comet for experiment tracking.

    This callback is adapted from the native Ray Comet callback and modified to monitor resource
    usage across all distributed workers during distributed training jobs. Note that it is not
    compatible with Ray Tune.

    Args:
        trainer: Ray Trainer object.
        tags: Tags to add to the logged Experiment.
        save_checkpoints: If `True`, model checkpoints will be saved to
            Comet ML as artifacts.
        share_api_key_to_workers: If `True`, Comet API key will be shared
            with workers via ray_config dictionary. This is an unsafe solution, and we recommend you
            to use a [more secure way to set up your API Key in your
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

        trainer = TorchTrainer(
            train_func,
            train_loop_config=config,
            scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
            run_config=RunConfig(callbacks=[comet_callback]),
        )
        comet_ray_train_logger(
            trainer=trainer,
            tags=["torch_ray_callback"],
            save_checkpoints=True,
            share_api_key_to_workers=True
        )

        result = trainer.fit()
        ```

    """
    train_loop_config = trainer._train_loop_config
    if train_loop_config is None:
        train_loop_config = {}
        trainer._train_loop_config = train_loop_config

    run_config = trainer.run_config
    if run_config.callbacks is not None and _has_comet_callback(run_config.callbacks):
        return

    callback = CometTrainLoggerCallback(
        ray_config=train_loop_config,
        tags=tags,
        save_checkpoints=save_checkpoints,
        share_api_key_to_workers=share_api_key_to_workers,
        experiment_name=experiment_name,
        api_key=api_key,
        workspace=workspace,
        project_name=project_name,
        experiment_key=experiment_key,
        mode=mode,
        online=online,
        **experiment_kwargs
    )

    if run_config.callbacks is None:
        run_config.callbacks = [callback]
    else:
        run_config.callbacks.append(callback)


def _has_comet_callback(callbacks: List[Any]) -> bool:
    for callback in callbacks:
        if hasattr(callback, "_setup_shared_comet_experiment"):
            return True

    return False
