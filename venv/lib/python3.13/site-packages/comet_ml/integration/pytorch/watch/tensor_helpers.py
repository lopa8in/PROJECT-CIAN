# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2021 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
import logging
from typing import TYPE_CHECKING, Optional

from comet_ml import CometExperiment
from comet_ml.logging_messages import (
    EXPERIMENT_PYTORCH_WATCH_DISTRIBUTED_TENSOR_NOT_SUPPORTED,
)

if TYPE_CHECKING:
    import numpy as np

import torch

LOGGER = logging.getLogger(__name__)

TENSOR_WARNING_PRINTED = False


def log_tensor_as_histogram(
    tensor: torch.Tensor, name: str, step: int, experiment: CometExperiment
) -> None:
    data = to_numpy(tensor)
    if data is None:
        return

    experiment.log_histogram_3d(
        data.flatten(),
        name=name,
        step=step,
    )


def to_numpy(x: torch.Tensor) -> Optional["np.ndarray"]:
    if not hasattr(x, "to_local"):
        # we check for attribute to_local which is only present in DTensor rather than do check isinstance(x, DTensor)
        # it is done for backward compatibility with PyTorch versions which doesn't have DTensor
        return _to_numpy(x)
    else:
        global TENSOR_WARNING_PRINTED
        if not TENSOR_WARNING_PRINTED:
            LOGGER.warning(EXPERIMENT_PYTORCH_WATCH_DISTRIBUTED_TENSOR_NOT_SUPPORTED)
            TENSOR_WARNING_PRINTED = True

        return None


def _to_numpy(x: torch.Tensor) -> "np.ndarray":
    x = x.detach()  # detach the tensor from computational graph to not corrupt it
    x = x.cpu()  # make sure the tensor located in CPU, not GPU/TPU memory
    x = x.to(
        torch.float32
    )  # make sure we end up with fp32 because pytorch supports different precisions
    x = x.numpy()
    return x
