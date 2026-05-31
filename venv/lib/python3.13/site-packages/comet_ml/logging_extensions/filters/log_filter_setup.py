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

from ...rate_limiter.duration import Duration, TimeUnit
from ...rate_limiter.limiter import OpRateLimiter
from ...rate_limiter.rate import OpRate
from ..callback import BaseLoggingCallback
from .log_filter import LogRecordRuleFilter
from .log_filter_helpers import type_from_extra_dict
from .log_record_types import ASSETS_UPLOAD_THROTTLING_MSG_EXTRA
from .log_rules import (
    CompositeAndLogRecordRule,
    LogRecordCallbackRule,
    RateLimitedLogRecordRule,
)


def setup_filters(logger: logging.Logger, logging_callback: BaseLoggingCallback):
    # once in 20 seconds rule
    once_in_20_throttling_rule = RateLimitedLogRecordRule(
        rate_limiter=OpRateLimiter([OpRate(1, Duration(20, TimeUnit.SECOND))]),
        drop_excess_records=True,
    )

    # log records callback rule
    records_callback = LogRecordCallbackRule(logging_callback)

    # setup  filter
    log_filter = LogRecordRuleFilter(logger.name)

    # add custom rules for different record types
    log_filter.add_rule(
        CompositeAndLogRecordRule([once_in_20_throttling_rule, records_callback]),
        record_type=type_from_extra_dict(ASSETS_UPLOAD_THROTTLING_MSG_EXTRA),
    )

    # add to the handlers
    for handler in logger.handlers:
        handler.addFilter(log_filter)
