# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2023 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************

import logging
import sys

from comet_ml import logging_messages
from comet_ml.logging_extensions import rich_decoration

LOGGER = logging.getLogger(__name__)


def present_pytorch_integration_log_model() -> None:
    if rich_decoration.is_available():
        try:
            rich_decoration.print_rule(logging_messages.NEW_FEATURE_PANEL_TITLE)
            print(
                logging_messages.PYTORCH_INTEGRATION_LOG_MODEL_RICH_ANNOUNCEMENT,
                file=sys.stderr,
            )
            rich_decoration.print_rule()
            return
        except Exception:
            pass

    LOGGER.info(logging_messages.PYTORCH_INTEGRATION_LOG_MODEL_ANNOUNCEMENT)
