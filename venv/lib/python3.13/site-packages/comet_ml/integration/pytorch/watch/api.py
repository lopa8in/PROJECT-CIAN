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
from typing import List

import comet_ml
from comet_ml import exceptions
from comet_ml.logging_messages import (
    EXPERIMENT_LOG_HOOK_REGISTRATION_WARNING,
    EXPERIMENT_LOG_NON_TRAINABLE_LAYER_INFO,
)

import torch.nn

from . import parameter_after_backward_hook, tensor_helpers

LOGGER = logging.getLogger(__name__)


def watch(model: torch.nn.Module, log_step_interval: int = 1000) -> None:
    """
    Enables automatic logging of each layer's parameters and gradients in the given PyTorch module.
    These will be logged as histograms. Note that an Experiment must be created before
    calling this function.

    Args:
        model: An instance of `torch.nn.Module`.
        log_step_interval: Determines how often layers are logged (default is every
            1000 steps).
    """
    experiment = comet_ml.get_global_experiment()

    if experiment is None:
        raise exceptions.CometException(
            "An Experiment must be created before calling `comet_ml.integration.pytorch.watch`"
        )

    initial_step = 0 if experiment.curr_step is None else experiment.curr_step
    params_non_trainable: List[str] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            params_non_trainable.append(name)
            continue

        after_backward_hook = parameter_after_backward_hook.ParameterAfterBackwardHook(
            experiment=experiment,
            name=name,
            parameter=parameter,
            log_step_interval=log_step_interval,
            initial_step=initial_step,
        )

        try:
            parameter.register_hook(after_backward_hook)
        except Exception:
            LOGGER.warning(
                EXPERIMENT_LOG_HOOK_REGISTRATION_WARNING.format(name),
                exc_info=True,
            )
            continue

        try:
            tensor_helpers.log_tensor_as_histogram(
                tensor=parameter,
                name=name,
                step=initial_step,
                experiment=experiment,
            )
        except Exception:
            LOGGER.debug(
                "failed to log histogram for %s at step %d",
                name,
                initial_step,
                exc_info=True,
            )

    if len(params_non_trainable) == 0:
        return

    message = [f"\n    {p}" for p in sorted(params_non_trainable)]
    message = "".join(message)
    LOGGER.info(EXPERIMENT_LOG_NON_TRAINABLE_LAYER_INFO.format(message))
