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
from typing import Callable, NamedTuple

from comet_ml.messages import BaseMessage


class ErrorReport(NamedTuple):
    message: str
    has_crashed: bool = False


class HandlerContext:
    """
    Represents the context for handling message loop and callback execution.

    This class provides the essential attributes necessary for managing the state
    of a message loop and executing specific callbacks for handling messages
    and errors. It is designed to be used in systems requiring a structured way
    to pass these handlers and states.

    Attributes:
        push_back_callback: A callable function used to push back messages
            into the system. The function must accept a parameter of type
            BaseMessage.
        report_error_callback: A callable function used for error reporting. The
            function must accept a parameter of type ErrorReport.
    """

    def __init__(
        self,
        message_loop_active: bool,
        push_back_callback: Callable[[BaseMessage], None],
        report_error_callback: Callable[[ErrorReport], None],
    ):
        self._message_loop_active = message_loop_active
        self.push_back_callback = push_back_callback
        self.report_error_callback = report_error_callback

    @property
    def message_loop_active(self):
        """
        Indicates whether the message loop is currently active.

        This property determines if the message loop, typically used
        for handling events, is running at the moment. It is useful
        for checking the status of event-driven systems.

        Returns:
            bool: True if the message loop is active, False otherwise.
        """
        return self._message_loop_active
