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
import threading
from typing import AnyStr, Callable, Optional

from ..config import DEFAULT_WAIT_FOR_FINISH_SLEEP_INTERVAL, PROGRESS_CALLBACK_INTERVAL
from ..logging_messages import (
    OFFLINE_SENDER_FAILED_TO_WRITE_ALL_DATA,
    OFFLINE_SENDER_REMAINING_DATA_ITEMS_TO_WRITE,
    OFFLINE_SENDER_WAIT_FOR_FINISH_PROMPT,
    STREAMER_UNKNOWN_MESSAGE_TYPE_EXCEPTION,
    UNEXPECTED_OFFLINE_STREAMER_ERROR,
)
from ..messages import BaseMessage, CloseMessage
from ..utils import log_once_at_level, wait_for_done
from .handlers.handler_context import HandlerContext
from .handlers.offline_handler import OfflineMessageHandler
from .progress_helpers import FixedIntervalProgressTracker

LOGGER = logging.getLogger(__name__)

from .base import BaseStreamer


class OfflineStreamer(BaseStreamer):
    """
    Handles streaming for offline experiments by writing messages to a local file.

    This class is designed for use in scenarios where data cannot be streamed to a
    remote server and must instead be stored locally for later retrieval or analysis.
    The OfflineStreamer provides message processing and serialization to files, ensuring
    that the generated data is correctly stored even in fallback or offline modes. It
    operates as a daemon thread to minimize its interference with the main process.

    Attributes:
        tmp_dir: Directory used to store temporary files corresponding to the
            messages being processed.
        wait_timeout: Maximum amount of time in seconds the streamer waits for
            remaining data to be processed during shutdown.
        on_error: Callback function invoked when an
            error occurs in the experiment. Defaults to None.
        fallback_mode: Indicates whether the streamer is running in fallback mode
            where it acts as a backup for online streaming.
    """

    def __init__(
        self,
        tmp_dir: AnyStr,
        wait_timeout: int,
        on_error_callback: Callable[[str], None] = None,
        fallback_mode: bool = False,
        experiment_key: Optional[str] = None,
        progress_callback_interval: float = PROGRESS_CALLBACK_INTERVAL,
        wait_for_finish_sleep_interval: float = DEFAULT_WAIT_FOR_FINISH_SLEEP_INTERVAL,
    ) -> None:
        super().__init__(initial_offset=0, queue_timeout=1)

        self.daemon = True
        if experiment_key is not None:
            self.name = "OfflineStreamer(experiment=%r)" % experiment_key

        self.tmp_dir = tmp_dir
        self.wait_timeout = wait_timeout
        self.on_error = on_error_callback
        self.fallback_mode = fallback_mode

        self._progress_callback_interval = progress_callback_interval
        self._wait_for_finish_sleep_interval = wait_for_finish_sleep_interval

        self.abort_processing = False
        self.__write_lock__ = threading.RLock()

        self.message_handler = OfflineMessageHandler(
            tmp_dir=tmp_dir,
            wait_timeout=wait_timeout,
            on_error_callback=on_error_callback,
            fallback_mode=fallback_mode,
            write_lock=self.__write_lock__,
        )

    def _after_run(self) -> None:
        # Close the messages files once we are sure we won't write in it
        # anymore
        self.message_handler.close()

    def _current_handler_context(self) -> HandlerContext:
        return HandlerContext(
            message_loop_active=self.is_message_loop_active(),
            push_back_callback=self.put_message_in_q,
            report_error_callback=lambda report: self._report_experiment_error(
                report.message, report.has_crashed
            ),
        )

    def is_message_loop_active(self) -> bool:
        return self.abort_processing is False

    def _loop(self) -> Optional[BaseMessage]:
        """
        A single loop of running
        """
        if self.abort_processing:
            # force stop the thread run()
            return CloseMessage()

        try:
            messages = self.getn(1)

            if messages is not None:
                LOGGER.debug(
                    "OfflineStreamer: Got %d messages, %d still in queue",
                    len(messages),
                    self.messages.qsize(),
                )

                for message in messages:
                    if isinstance(message, CloseMessage):
                        return message
                    elif self.abort_processing:
                        # ignore all messages to empty the queue
                        continue

                    if self.message_handler.has_handler(message):
                        self.message_handler.handle(
                            message, self._current_handler_context()
                        )
                    else:
                        raise ValueError(
                            STREAMER_UNKNOWN_MESSAGE_TYPE_EXCEPTION, message
                        )

        except Exception as ex:
            LOGGER.warning(UNEXPECTED_OFFLINE_STREAMER_ERROR, ex, exc_info=True)
            self._report_experiment_error(UNEXPECTED_OFFLINE_STREAMER_ERROR % ex)

        return None

    def _report_experiment_error(self, message: str, has_crashed: bool = False) -> None:
        if self.on_error is not None:
            self.on_error(message)

    def _abort_processing(self, timeout: float = 10) -> None:
        self.abort_processing = True

        if self.__write_lock__.acquire(timeout=timeout):
            # make sure to wait for the current write operation (if any in progress) to complete within the given timeout
            self.__write_lock__.release()

    def abort_and_wait(self, timeout: float = 10, _join: bool = True) -> None:
        """Aborts streamer by forcing an immediate drop in processing of all scheduled messages. The currently
        processed message can still be written to the disk. If there is a message currently in process of writing
        to the disk, the invoking thread will be blocked until message data is fully written. Otherwise, this method
        waits for the offline streamer's thread to terminate and return."""
        if not self.is_alive():
            return

        self._abort_processing()

        if _join is True:
            self.join(timeout)

    def wait_for_finish(self, timeout: float = 10, _join: bool = True) -> bool:
        """Blocks the experiment from exiting until all data is saved or timeout exceeded."""
        if not self.is_alive():
            # already finished
            return True

        wait_for_done(
            lambda: self.closed,
            timeout=self.wait_timeout,
            sleep_time=self._wait_for_finish_sleep_interval,
        )

        log_once_at_level(
            logging.INFO,
            OFFLINE_SENDER_WAIT_FOR_FINISH_PROMPT,
            int(self.wait_timeout * 2),
        )

        # Wait maximum for 2 times of self.wait_timeout
        progress_callback = FixedIntervalProgressTracker(
            interval=self._progress_callback_interval,
            progress_callback=lambda: LOGGER.info(
                OFFLINE_SENDER_REMAINING_DATA_ITEMS_TO_WRITE, self.messages.qsize()
            ),
        )
        wait_for_done(
            check_function=self.messages.empty,
            timeout=self.wait_timeout,
            progress_callback=progress_callback,
            sleep_time=self._wait_for_finish_sleep_interval,
        )

        if not self.messages.empty():
            LOGGER.info(OFFLINE_SENDER_WAIT_FOR_FINISH_PROMPT, int(self.wait_timeout))
            if not self.messages.empty():
                wait_for_done(
                    check_function=self.messages.empty,
                    timeout=self.wait_timeout,
                    progress_callback=progress_callback,
                    sleep_time=self._wait_for_finish_sleep_interval,
                )

        # stop message loop after all flush attempts
        self._abort_processing()

        if not self.messages.empty():
            remaining = self.messages.qsize()
            LOGGER.error(OFFLINE_SENDER_FAILED_TO_WRITE_ALL_DATA, remaining)
            self._report_experiment_error(
                OFFLINE_SENDER_FAILED_TO_WRITE_ALL_DATA % remaining
            )

            self.abort_and_wait(timeout)
            return False

        # Also wait for the thread to finish to be sure that all messages are
        # written to the message file
        if _join:
            self.join(timeout)

        LOGGER.debug("OfflineStreamer finished in time")
        return True

    def has_connection_to_server(self) -> bool:
        return False

    def flush(self, timeout: Optional[int] = None) -> bool:
        return True
