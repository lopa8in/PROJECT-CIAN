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
import abc
from typing import Callable, Dict, Optional, Type

from ...logging_messages import STREAMER_UNKNOWN_MESSAGE_TYPE_EXCEPTION
from ...messages import BaseMessage
from .handler_context import HandlerContext

MessageHandlerType = Callable[[BaseMessage, HandlerContext], Optional[bool]]


class BaseMessageHandler(abc.ABC):
    """
    BaseMessageHandler is an abstract base class for handling messages and delegating their processing
    to appropriate handlers.

    This class is designed to manage a collection of message handlers that are linked to specific
    message types. It provides core functionality for identifying the type of incoming message
    and delegating it to a corresponding handler, while also offering a contract to implement message
    handler creation. The purpose of this class is to ensure consistent and extendable message handling
    mechanisms in derived classes.
    """

    def __init__(self):
        self.message_handlers = self.create_message_handlers()

    def handle(self, message: BaseMessage, context: HandlerContext) -> Optional[bool]:
        """
        Handles incoming messages and delegates them to appropriate message handlers.

        This method identifies the type of the incoming message and finds the corresponding handler
        from the registered message handlers. If no handler is found for the given message type,
        an exception is raised. Otherwise, the message is processed using the identified handler.

        Args:
        message: BaseMessage
            The incoming message to handle, whose type will determine the appropriate handler.
        context: HandlerContext
            The context in which the message should be processed, providing additional information
            or state.

        Returns:
        bool
            Indicates whether the message was successfully handled. If None returned, the message
            status cannot be determined immediately due to asynchronous processing or other reasons.

        Raises:
        ValueError
            If the type of the incoming message is unrecognized.
        """
        handler = self.message_handlers.get(type(message))
        if handler is None:
            raise ValueError(STREAMER_UNKNOWN_MESSAGE_TYPE_EXCEPTION, message)
        return handler(message, context)

    def has_handler(self, message: BaseMessage) -> bool:
        """
        Checks if there is a handler for the provided message type.

        This method determines if the type of the input message exists in
        the message handlers dictionary. It is used to check the availability
        of a corresponding handler for the given message.

        Args:
            message (BaseMessage): The message whose type is to be checked
            against the registered handlers.

        Returns:
            bool: True if a handler for the message type exists, otherwise False.
        """
        return type(message) in self.message_handlers

    @abc.abstractmethod
    def create_message_handlers(self) -> Dict[Type[BaseMessage], MessageHandlerType]:
        """
        Defines an abstract method for creating message handlers. This method must be
        implemented by subclasses to provide specific mappings between message
        types and their respective handlers.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Returns:
            Dict[Type[BaseMessage], MessageHandlerType]: A dictionary mapping message
            types to their corresponding handler functions or callable entities.
        """
        raise NotImplementedError()
