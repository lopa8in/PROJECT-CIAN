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

import copy
import logging


class CometFileFormatter(logging.Formatter):
    REDACTED_STRING = "*" * 9 + "REDACTED" + "*" * 9

    def __init__(self, *args, **kwargs):
        hide_traceback = kwargs.pop("hide_traceback", False)
        super().__init__(*args, **kwargs)

        self.hide_traceback = hide_traceback
        self.strings_to_redact = set()

    def format(self, record: logging.LogRecord) -> str:
        if (
            getattr(record, "show_traceback", False) is False
            and self.hide_traceback is True
        ):
            # Make a copy of the record to avoid altering it
            new_record = copy.copy(record)

            # And delete exception information so no traceback could be formatted
            # and displayed
            new_record.exc_info = None
            new_record.exc_text = None
        else:
            new_record = record

        result = super().format(new_record)
        # If s is not in result, it's faster to check first before doing the substring replacement
        for s in self.strings_to_redact:
            # Avoid redacting strings if they are passed explicitly as log records
            if s in result and s not in record.args:
                result = result.replace(s, self.REDACTED_STRING)
        return result
