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
from .filter_chain import FilterChain
from .filter_count_handler import FilterCountHandler
from .filter_invalid_api_key_handler import FilterInvalidAPIKeyHandler
from .filter_throttling_handler import FilterThrottlingHandler

DEFAULT_ERROR_QUOTA = 50
DEFAULT_WARNING_QUOTA = 50

filter_invalid_api_key_handler = FilterInvalidAPIKeyHandler()
filter_throttling_handler = FilterThrottlingHandler()
filter_error_count_handler = FilterCountHandler(
    max_count=DEFAULT_ERROR_QUOTA, level="error"
)
filter_warning_count_handler = FilterCountHandler(
    max_count=DEFAULT_WARNING_QUOTA, level="warning"
)


chain = [
    filter_invalid_api_key_handler,
    filter_throttling_handler,
    filter_error_count_handler,
    filter_warning_count_handler,
]

sentry_filter_chain = FilterChain(handlers=chain)
