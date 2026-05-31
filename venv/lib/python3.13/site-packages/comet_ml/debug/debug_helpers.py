# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.com
#  Copyright (C) 2015-2024 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
import logging
from contextlib import contextmanager
from typing import Any, Callable

from .._logging import _get_comet_logging_config
from ..exceptions import CometDebugException
from .debug_options import (
    _enable_debug_exception_raising,
    has_enabled_debug_exception_raising,
)

LOGGER = logging.getLogger(__name__)


def error_mode() -> None:
    """Enables the printing of error tracebacks to the console logger for debugging purposes."""
    try:
        _get_comet_logging_config().enable_console_traceback()
    except Exception:
        LOGGER.warning("Failed to enable console traceback", exc_info=True)

    _enable_debug_exception_raising()


def _debug_error_handler_executor(f: Callable, *args, **kwargs) -> Any:
    return f(*args, **kwargs)


@contextmanager
def debug_error_handler(
    message: str, logger: logging.Logger, exc_info: bool = True
) -> Callable:
    """The context manager returns an executor that is protected by an exception handling block.
    If an exception is raised by the wrapped user function, it will:
    - Raise an original exception if comet_ml.error_mode() is enabled.
    - Otherwise, log an ERROR message to the provided logger."""
    try:
        yield _debug_error_handler_executor
    except Exception as e:
        if has_enabled_debug_exception_raising():
            raise e
        else:
            logger.error(msg=message, exc_info=exc_info)


def raise_debug_exception(message: str, *args: Any) -> None:
    if args and len(args) > 0:
        message = message % args
    raise CometDebugException(message)


def raise_debug_exception_or_ignore(message: str, *args: Any) -> None:
    """Raises a CometDebugException if comet_ml.error_mode() is enabled. Otherwise - just do nothing."""
    if has_enabled_debug_exception_raising():
        raise_debug_exception(message, *args)


def log_warning_or_raise(message: str, *args, **kwargs) -> None:
    """Raises a ``CometDebugException`` or ``original_exception`` if ``comet_ml.error_mode()`` is enabled.
    Otherwise - just log the provided ``message`` into specified ``logger`` or ``LOGGER`` as a warning.

    Args:
        message: the message using ``%``-formatting
        args: the message format arguments
        original_exception: the original exception if any
        logger: the logger to be used for message output
        exc_info: whether to display backtrace of exception

    log_warning_or_raise("We have a %s", "mysterious problem", original_exception=e, logger=LOGGER, exc_info=True)
    """
    _log_or_raise(logging.WARNING, message, *args, **kwargs)


def log_error_or_raise(message: str, *args, **kwargs) -> None:
    """Raises a ``CometDebugException`` or ``original_exception`` if ``comet_ml.error_mode()`` is enabled.
    Otherwise - just log the provided ``message`` into specified ``logger`` or ``LOGGER`` as an error.

    Args:
        message: the message using ``%``-formatting
        args: the message format arguments
        original_exception: the original exception if any
        logger: the logger to be used for message output
        exc_info: whether to display backtrace of exception

    log_error_or_raise("We have a %s", "mysterious problem", original_exception=e, logger=LOGGER, exc_info=True)
    """
    _log_or_raise(logging.ERROR, message, *args, **kwargs)


def _log_or_raise(level: int, message: str, *args, **kwargs) -> None:
    original_exception = kwargs.pop("original_exception", None)
    if has_enabled_debug_exception_raising():
        if original_exception is not None:
            raise original_exception
        else:
            raise_debug_exception(message, *args)
    else:
        logger = kwargs.pop("logger", None) if not None else LOGGER
        logger.log(level, message, *args, **kwargs)
