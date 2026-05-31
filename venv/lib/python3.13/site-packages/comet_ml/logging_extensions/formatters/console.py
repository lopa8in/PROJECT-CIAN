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

import functools
import logging
import sys
from typing import Mapping, Optional, Tuple, Union

from comet_ml import _jupyter

from .. import rich_decoration
from ..rich_decoration import loglevel_colors
from . import file, url


class CometConsoleFormatter(file.CometFileFormatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._rich_available = rich_decoration.is_available()
        self._in_jupyter = _jupyter._in_jupyter_environment()

    def format(self, record: logging.LogRecord) -> str:
        record.prefix = self._get_prefix(record.levelname)
        record.args = self._processed_arguments(record.args)

        return super().format(record)

    def enable_traceback(self) -> None:
        self.hide_traceback = False

    def _get_prefix(self, levelname: str) -> str:
        prefix = "COMET {}:".format(levelname)

        if self._rich_available:
            color = loglevel_colors.get(levelname)
            stylized_prefix = _stylized_text(prefix, color, bold=True)
            prefix = prefix if stylized_prefix is None else stylized_prefix

        return prefix

    def _processed_arguments(
        self, arguments: Union[Mapping, Tuple, None]
    ) -> Union[Mapping, Tuple, None]:
        if self._in_jupyter or not self._rich_available:
            return arguments

        if not isinstance(arguments, tuple):
            return arguments

        result = []

        for argument in arguments:
            if url.is_url(argument):
                stylized_url = _stylized_text(argument, "deep_sky_blue1")
                argument = argument if stylized_url is None else url.URL(stylized_url)

            result.append(argument)

        return tuple(result)


@functools.lru_cache(maxsize=0 if "pytest" in sys.modules else 256)
def _stylized_text(text: str, color: str, **kwargs) -> Optional[str]:
    try:
        return rich_decoration.stylize_text(text, color, **kwargs)
    except Exception:
        return None
