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
from typing import Optional

from comet_ml import exceptions, logging_messages
from comet_ml.secrets import interpreter

LOGGER = logging.getLogger(__name__)


def perform_checks(api_key: Optional[str]):
    if api_key is None:
        raise exceptions.CometException(
            "Comet.ml requires an API key. "
            "Please provide it as an argument to CometVertexPipelineLogger(api_key) "
            "or as an environment variable named COMET_API_KEY"
        )

    if not interpreter.is_secret_instruction(api_key):
        LOGGER.warning(logging_messages.INSECURE_KEY_SHARING_VERTEX)
