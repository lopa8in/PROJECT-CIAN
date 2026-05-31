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
from logging import Filter, LogRecord
from typing import Dict, Optional

from .log_filter_helpers import _EXTRA_LOG_RECORD_TYPE_ATTR
from .log_rules import LogRecordRule


class LogRecordRuleFilter(Filter):

    def __init__(self, name: str):
        super().__init__(name)
        self.rules: Dict[str, LogRecordRule] = {}

    def filter(self, record: LogRecord) -> bool:
        if hasattr(record, _EXTRA_LOG_RECORD_TYPE_ATTR):
            record_type = getattr(record, _EXTRA_LOG_RECORD_TYPE_ATTR)
            return self._apply_rules(record, record_type=record_type) is not None

        return True

    def add_rule(self, rule: LogRecordRule, record_type: str):
        if record_type in self.rules:
            raise ValueError(f"Rule for type: '{record_type}' already registered")
        self.rules[record_type] = rule

    def _apply_rules(self, record: LogRecord, record_type: str) -> Optional[LogRecord]:
        if record_type not in self.rules:
            # just return unprocessed record
            return record

        return self.rules[record_type].apply(record, record_type=record_type)
