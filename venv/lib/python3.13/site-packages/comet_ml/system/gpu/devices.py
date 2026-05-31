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
import os
from typing import List, Optional

from comet_ml.vendor.nvidia_ml import pynvml

from .. import processes
from . import cuda, identifiers
from .cuda import _NO_DEVICE_VALUES

LOGGER = logging.getLogger(__name__)


def find_visible() -> List[str]:
    try:
        pynvml.nvmlInit()

        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            LOGGER.debug("CUDA_VISIBLE_DEVICES is unset, defaulting to all devices")
            return identifiers.all()

        cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]

        if cuda_visible_devices in _NO_DEVICE_VALUES:
            LOGGER.debug(
                "CUDA_VISIBLE_DEVICES is set to %s, no GPUs will be used"
                % cuda_visible_devices
            )

        uuids = cuda.parse_visible_devices(cuda_visible_devices)
        LOGGER.debug("CUDA_VISIBLE_DEVICES uuids={}".format(uuids))
    except pynvml.NVMLError:
        LOGGER.debug(
            "An internal pynvml error was caught while getting visible devices list. Returning None",
            exc_info=True,
        )
        uuids = None

    return uuids


def find_used() -> Optional[List[str]]:
    if not processes.is_available():
        return None

    gpus_consumers = processes.gpus_consumers()
    processes_family = set(processes.family())

    used_gpus = []
    for uuid, consumers in gpus_consumers.items():
        used_by_us = len(set(consumers).intersection(processes_family)) > 0
        if used_by_us:
            used_gpus.append(uuid)

    return sorted(used_gpus)


def all():
    return identifiers.all()
