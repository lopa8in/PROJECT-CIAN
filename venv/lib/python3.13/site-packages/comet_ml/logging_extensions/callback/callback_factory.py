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
from typing import List

from .base_callback import BaseLoggingCallback
from .throttling_error_callback import ThrottlingErrorCallback


class CallbackFactory(BaseLoggingCallback):

    def __init__(self, callbacks: List[BaseLoggingCallback] = None):
        self.callbacks = [ThrottlingErrorCallback()]
        if callbacks is not None:
            self.callbacks.extend(callbacks)

    def on_log_record(self, record: logging.LogRecord, record_type: str) -> None:
        for callback in self.callbacks:
            callback.on_log_record(record, record_type)

    def flush(self, logger: logging.Logger) -> None:
        for callback in self.callbacks:
            callback.flush(logger)
