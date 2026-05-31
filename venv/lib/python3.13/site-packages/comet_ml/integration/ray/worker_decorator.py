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
import functools
import logging

from .worker_logger import comet_worker_logger

LOGGER = logging.getLogger(__file__)


def comet_worker(func):
    """
    This decorator enables you to monitor resource usage for each distributed worker during a distributed training job.
    By applying this decorator, you can annotate any training function to integrate Comet’s resource tracking.

    Note: This should be used together with the [comet_ml.integration.ray.CometTrainLoggerCallback][] callback,
    and the training function must accept a configuration dictionary as an input argument.

    Args:
        func (Callable): The training function to be wrapped which should have configuration dictionary as an input argument.
            The training function is a user-defined Python function that contains the end-to-end model training
            loop logic. When launching a distributed training job, each worker executes this training function.

    Example:
        ```python linenums="1"
        @comet_worker
        def train_func(config: Dict):
            # ray worker training code
        ```
    """

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        if len(args) == 1:
            config = args[0]
        else:
            config = {}
            LOGGER.warning(
                "The training function is missing a configuration dictionary as an input argument. "
                "As a result, Comet resource usage monitoring will be disabled. "
                "Please use a training function that includes an input argument."
            )

        with comet_worker_logger(ray_config=config):
            func(*args, **kwargs)

    return wrapper_decorator
