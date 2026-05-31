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
import logging
import threading
from abc import ABC
from queue import Empty, Queue
from typing import List, Optional

from comet_ml.logging_messages import STREAMER_CLOSED_PUT_MESSAGE_FAILED
from comet_ml.messages import BaseMessage, CloseMessage

LOGGER = logging.getLogger(__name__)


class BaseStreamer(threading.Thread, ABC):
    """
    Represents a base thread class for managing a message streaming queue.

    This class serves as an abstract base for custom streamers and provides the foundation to
    maintain a queue of messages, handle message processing, and manage the lifecycle of the
    streaming process. It should be extended and implemented with the required abstract methods.

    Attributes:
        messages: A thread-safe queue to hold and manage messages.
        queue_timeout: Timeout duration in seconds for queue operations.
        closed: A flag to indicate if the streamer has been closed, stopping further operations.
    """

    def __init__(self, initial_offset: int, queue_timeout: float) -> None:
        threading.Thread.__init__(self)

        self._counter = initial_offset
        self.messages = Queue()
        self.queue_timeout = queue_timeout
        self.closed = False
        self.__lock_closed__ = threading.RLock()

        LOGGER.debug("%r instantiated with duration %s", self, self.queue_timeout)

    def put_message_in_q(self, message: BaseMessage):
        """
        Puts a message in the queue
        :param message: Some kind of payload, type agnostic
        """
        with self.__lock_closed__:
            if message is not None:
                self._counter += 1
                if message.message_id is None:
                    # update message_id only if it was not already set
                    message.message_id = self._counter

                if not self.closed:
                    self.messages.put(message)
                else:
                    LOGGER.debug(STREAMER_CLOSED_PUT_MESSAGE_FAILED)
                    LOGGER.debug("Ignored message (streamer closed): %s", message)

    def close(self) -> None:
        """
        Marks the streamer as closed, stopping further operations.
        """
        with self.__lock_closed__:
            if self.closed:
                LOGGER.debug("Streamer: %r tried to be closed more than once", self)
                return

            # mark as closed to block any new messages
            self.closed = True

            LOGGER.debug("Streamer %r, closed: %r", self, self.closed)

    def stop(self) -> None:
        """
        Stops the message loop for the system by enqueuing a CloseMessage, which signals
        the termination of the message processing sequence.

        This method is generally used to terminate a running message loop gracefully.
        """
        LOGGER.debug("Gracefully stopping the message loop of the streamer %r", self)

        self._put_close_message_in_queue()

    def _put_close_message_in_queue(self) -> None:
        self.messages.put(CloseMessage())

    def _before_run(self) -> None:
        pass

    def run(self) -> None:
        """
        Continuously pulls messages from the queue and process them.
        """
        self._before_run()

        while True:
            out = self._loop()

            # Exit the infinite loop
            if out is not None and isinstance(out, CloseMessage):
                break

        self._after_run()

        LOGGER.debug(
            "%r has finished RUN loop, closed: %r", self.__class__, self.closed
        )

    @abc.abstractmethod
    def wait_for_finish(self, **kwargs) -> bool:
        pass

    @abc.abstractmethod
    def has_connection_to_server(self) -> bool:
        pass

    @abc.abstractmethod
    def flush(self, timeout: Optional[int] = None) -> bool:
        pass

    @abc.abstractmethod
    def _loop(self) -> Optional[BaseMessage]:
        pass

    @abc.abstractmethod
    def _report_experiment_error(self, message: str, has_crashed: bool = False) -> None:
        pass

    def _after_run(self):
        pass

    def getn(self, n: int) -> Optional[List[BaseMessage]]:
        """
        Pops n messages from the queue.
        Args:
            n: Number of messages to pull from queue

        Returns: n messages

        """
        try:
            msg = self.messages.get(
                timeout=self.queue_timeout
            )  # block until at least 1
        except Empty:
            LOGGER.debug("No message in queue, timeout")
            return None

        if isinstance(msg, CloseMessage):
            return [msg]

        result = [msg]
        try:
            while len(result) < n:
                another_msg = self.messages.get_nowait()
                result.append(another_msg)
        except Empty:
            LOGGER.debug("No more messages in queue, while getting more than 1 message")
        return result
