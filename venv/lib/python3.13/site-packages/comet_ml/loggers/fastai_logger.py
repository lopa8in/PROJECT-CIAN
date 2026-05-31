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

from .. import logging_messages
from ..monkey_patching import check_module

LOGGER = logging.getLogger(__name__)
FRAMEWORK = "fastai"


def learner_constructor(experiment, original, *args, **kwargs):
    try:
        comet_callback = experiment.get_callback(FRAMEWORK)
    except Exception:
        LOGGER.warning(logging_messages.GET_CALLBACK_FAILURE, FRAMEWORK, exc_info=True)
        return None

    try:
        ## args[0]: Learner
        learner = args[0]
        is_comet_callback_present = any(
            isinstance(callback, type(comet_callback)) for callback in learner.cbs
        )

        if not is_comet_callback_present:
            learner.add_cb(comet_callback)

    except Exception:
        LOGGER.error("Failed to run Learner.fit logger", exc_info=True)


def patch(module_finder):
    ## For testing:
    check_module("fastai")

    module_finder.register_before("fastai.learner", "Learner.fit", learner_constructor)


check_module("fastai")
