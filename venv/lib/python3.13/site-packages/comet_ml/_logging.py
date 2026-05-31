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

"""
Author: Boris Feld

This module contains logging configuration for Comet

"""
import functools
import logging
import os
import re
from datetime import datetime
from typing import Callable, Optional

from . import logging_messages
from .constants import DEFAULT_PROJECT_NAME
from .logging_extensions import formatters
from .logging_extensions.callback import CallbackFactory
from .logging_extensions.filters import setup_filters
from .logging_messages import CONSOLE_DEBUG_DEPRECATED
from .utils import get_user

LOGGER = logging.getLogger(__name__)


def expand_log_file_path(log_file_path: str, project_name: Optional[str] = None) -> str:
    """
    Expand patterns in the file logging path.

    Allowed patterns:
        * {datetime}
        * {pid}
        * {project}
        * {user}
    """

    if project_name is None:
        project_name = DEFAULT_PROJECT_NAME

    def make_valid(s: str) -> str:
        s = str(s).strip().replace(" ", "_")
        return re.sub(r"(?u)[^-\w.]", "", s)

    user = make_valid(get_user())

    patterns = {
        "datetime": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "pid": os.getpid(),
        "project": project_name,
        "user": user,
    }
    if log_file_path is not None:
        try:
            log_file_path = log_file_path.format(**patterns)
        except KeyError:
            LOGGER.info(
                "Invalid logging file pattern: '%s'; ignoring",
                log_file_path,
                exc_info=True,
            )

    return log_file_path


class CometLoggingConfig(object):
    def __init__(self):
        self.root: Optional[logging.Logger] = None
        self.console_handler: Optional[logging.StreamHandler] = None
        self.console_formatter: Optional[formatters.CometConsoleFormatter] = None
        self.file_handler: Optional[logging.FileHandler] = None
        self.file_formatter: Optional[formatters.CometFileFormatter] = None
        self.logging_callback: Optional[CallbackFactory] = None

    def setup(self, config) -> logging.Logger:
        self.root = logging.getLogger("comet_ml")
        self.logging_callback = CallbackFactory()
        logger_level = logging.CRITICAL
        message_format = logging_messages.MSG_FORMAT

        file_message_format = logging_messages.FILE_MSG_FORMAT

        # Don't send comet-ml to the application logger
        self.root.propagate = False

        # Add handler for console, basic INFO:
        self.console_handler = logging.StreamHandler()
        logging_console = config["comet.logging.console"]

        if logging_console and logging_console.upper() in [
            "DEBUG",
            "ERROR",
            "INFO",
            "CRITICAL",
            "FATAL",
            "WARN",
            "WARNING",
        ]:
            logging_console_level = logging._checkLevel(logging_console.upper())
            hide_traceback = False
        else:
            logging_console_level = logging.INFO
            hide_traceback = True

        if logging_console_level == logging.DEBUG:
            LOGGER.warning(CONSOLE_DEBUG_DEPRECATED)
            logging_console_level = logging.INFO

        self.console_formatter = formatters.CometConsoleFormatter(
            message_format, hide_traceback=hide_traceback
        )

        self.console_handler.setLevel(logging_console_level)
        self.console_handler.setFormatter(self.console_formatter)
        self.root.addHandler(self.console_handler)

        logger_level = min(logger_level, self.console_handler.level)

        # The std* logger might conflict with the logging if a log record is
        # emitted for each WS message as it would result in an infinite loop. To
        # avoid this issue, all log records after the creation of a message should
        # be at a level lower than info as the console handler is set to info
        # level.

        # Add the additional file handler
        log_file_path = expand_log_file_path(
            config.get_log_file_path(), config["comet.project_name"]
        )
        log_file_level = config["comet.logging.file_level"]
        log_file_overwrite = config["comet.logging.file_overwrite"]

        self.file_handler = None
        self.file_formatter = None

        if log_file_path is not None:

            # Create logfile path, if possible:
            if os.path.dirname(log_file_path) != "":
                try:
                    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
                except Exception:
                    LOGGER.error(
                        "can't create path to log file %r", log_file_path, exc_info=True
                    )

            try:
                # Overwrite file if comet.logging.file_overwrite:
                if log_file_overwrite:
                    self.file_handler = logging.FileHandler(log_file_path, "w+")
                else:
                    self.file_handler = logging.FileHandler(log_file_path)

                if log_file_level is not None:
                    log_file_level = logging._checkLevel(log_file_level.upper())
                else:
                    log_file_level = logging.DEBUG

                self.file_handler.setLevel(log_file_level)
                logger_level = min(logger_level, log_file_level)

                if log_file_level == logging.DEBUG:
                    file_message_format = logging_messages.DEBUG_FILE_MSG_FORMAT

                self.file_formatter = formatters.CometFileFormatter(
                    file_message_format, hide_traceback=False
                )
                self.file_handler.setFormatter(self.file_formatter)
                self.root.addHandler(self.file_handler)
            except Exception:
                LOGGER.error(
                    "can't open log file %r; file logging is disabled",
                    log_file_path,
                    exc_info=True,
                )

        self.root.setLevel(logger_level)
        return self.root

    def redact_string(self, string: str) -> None:
        """Blacklists specified string from logging. The string will be masked in the logging."""
        if self.console_formatter:
            self.console_formatter.strings_to_redact.add(string)

        if self.file_formatter:
            self.file_formatter.strings_to_redact.add(string)

    def discard_string_from_redact(self, string: str) -> None:
        """Discards specified string from the set of blacklisted strings. After this method invocation the string will
        no longer be masked from the logger output.
        """
        if self.console_formatter:
            self.console_formatter.strings_to_redact.discard(string)

        if self.file_formatter:
            self.file_formatter.strings_to_redact.discard(string)

    def enable_console_traceback(self) -> None:
        self.console_formatter.enable_traceback()

    def flush_logging_callback(self) -> None:
        self.logging_callback.flush(self.root)


COMET_LOGGING_CONFIG = CometLoggingConfig()


def _get_comet_logging_config() -> CometLoggingConfig:
    return COMET_LOGGING_CONFIG


def _setup_comet_logging(config):
    logger = COMET_LOGGING_CONFIG.setup(config)
    setup_filters(logger, COMET_LOGGING_CONFIG.logging_callback)


def convert_exception_to_log_message(
    message: str,
    logger: logging.Logger,
    return_on_exception=None,
    exception_info: bool = True,
    logging_level: int = logging.ERROR,
) -> Callable:
    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception:
                logger.log(logging_level, message, exc_info=exception_info)
                return return_on_exception

        return wrapper

    return decorator
