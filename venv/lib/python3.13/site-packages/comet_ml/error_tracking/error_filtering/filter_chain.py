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
from typing import List

from comet_ml.logging_messages import SENTRY_FILTER_CHAIN_EXCEPTION

from ..types import Event, Hint
from .handler import Handler

LOGGER = logging.getLogger(__name__)


class FilterChain:
    def __init__(self, handlers: List[Handler]):
        self._handlers = handlers

    def validate(self, event: Event, hint: Hint) -> bool:
        try:
            for handler in self._handlers:
                if not handler.process_event(event, hint):
                    return False

            return True
        except Exception as e:
            LOGGER.debug(SENTRY_FILTER_CHAIN_EXCEPTION % e)

            return False
