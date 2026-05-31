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
from requests.exceptions import HTTPError

from ..types import Event, Hint
from .handler import Handler

TOO_MANY_REQUESTS_STATUS_CODE = 429


class FilterThrottlingHandler(Handler):
    def process_event(self, event: Event, hint: Hint) -> bool:
        if hint.get("exc_info", False):
            exc_type, exc_value, tb = hint["exc_info"]

            if (
                isinstance(exc_value, HTTPError)
                and exc_value.response is not None
                and exc_value.response.status_code == TOO_MANY_REQUESTS_STATUS_CODE
            ):
                return False

        return True
