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
from typing import Any, Dict

# the name of LogRecord attribute to hold our custom log record type
_EXTRA_LOG_RECORD_TYPE_ATTR = "log_rec_type"


def extra_for_type(record_type: str) -> Dict[str, Any]:
    return {_EXTRA_LOG_RECORD_TYPE_ATTR: record_type}


def type_from_extra_dict(extra: Dict[str, Any]) -> str:
    return extra[_EXTRA_LOG_RECORD_TYPE_ATTR]
