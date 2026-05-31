# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2025 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
import logging
import time
from typing import Optional

from comet_ml.config import DEFAULT_RETRY_AFTER_SECONDS

import requests


def extract_retry_after(response: requests.Response) -> Optional[float]:
    retry_after = response.headers.get("Retry-After-Custom")
    if retry_after:
        return float(retry_after)
    return None


def calculate_reset_at_from_response(
    response: requests.Response, logger: logging.Logger
) -> float:
    retry_after = extract_retry_after(response)
    if retry_after is None:
        logger.debug(
            "No Retry-After-Custom header found in response, using default value [%.1f sec]",
            DEFAULT_RETRY_AFTER_SECONDS,
        )
        retry_after = DEFAULT_RETRY_AFTER_SECONDS

    # add 0.5 sec to make sure we do not hit throttling immediately after reset due to rounding effects
    retry_after += 0.5
    return time.time() + retry_after
