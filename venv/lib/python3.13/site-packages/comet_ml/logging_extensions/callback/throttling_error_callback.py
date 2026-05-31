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
from typing import Set

from comet_ml.logging_extensions.callback.base_callback import BaseLoggingCallback


class ThrottlingErrorCallback(BaseLoggingCallback):
    def __init__(self):
        self.log_records: Set[str] = set()

    def on_log_record(self, record: logging.LogRecord, record_type: str) -> None:
        if record_type is not None and record_type.endswith("-throttling-msg"):
            self.log_records.add(record.getMessage())

    def flush(self, logger: logging.Logger) -> None:
        for record in self.log_records:
            logger.error(record)

        # clean log records
        self.log_records.clear()
