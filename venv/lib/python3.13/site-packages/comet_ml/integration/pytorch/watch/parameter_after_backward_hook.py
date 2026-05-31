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

from comet_ml import CometExperiment

import torch

from . import tensor_helpers

LOGGER = logging.getLogger(__name__)


class ParameterAfterBackwardHook:
    def __init__(
        self,
        experiment: CometExperiment,
        name: str,
        parameter: torch.nn.Parameter,
        log_step_interval: int,
        initial_step: int,
    ) -> None:
        self._experiment = experiment

        self._name = name
        self._log_step_interval = log_step_interval
        self._current_step = initial_step
        self._parameter = parameter

        self._gradients = 0  # it will become torch tensor after the first accumulation

    def __call__(self, grad: torch.Tensor) -> None:
        self._current_step += 1
        self._gradients += grad
        needs_logging = (self._current_step % self._log_step_interval) == 0

        if not needs_logging:
            return

        try:
            tensor_helpers.log_tensor_as_histogram(
                tensor=self._gradients,
                name=f"gradient/{self._name}",
                step=self._current_step,
                experiment=self._experiment,
            )

            tensor_helpers.log_tensor_as_histogram(
                tensor=self._parameter,
                name=self._name,
                step=self._current_step,
                experiment=self._experiment,
            )
        except Exception:
            LOGGER.debug(
                "Failed to log model weights and gradients for %s at step %d",
                self._name,
                self._current_step,
                exc_info=True,
            )

        self._gradients = 0
